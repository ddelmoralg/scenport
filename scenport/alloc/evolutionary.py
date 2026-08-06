"""GA, DE, PSO and CMA-ES over the encoded portfolio space.

Same decoder, same objective, same budget, so differences between them are
differences between the operators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import Allocation, Allocator, register
from .objective import Problem, decode, score_encoded

log = logging.getLogger(__name__)


@dataclass
class Budget:
    pop_size: int = 200
    generations: int = 300
    patience: int = 40
    tol: float = 1e-6


class PopulationSearch(Allocator):
    """Shared loop: initialise, iterate, stop early when the incumbent stalls."""

    stochastic = True
    init_scale = 0.5

    def budget(self) -> Budget:
        return Budget(
            pop_size=int(self.params.get("pop_size", 200)),
            generations=int(self.params.get("generations", 300)),
            patience=int(self.params.get("patience", 40)),
            tol=float(self.params.get("tol", 1e-6)),
        )

    def initialise(self, problem: Problem, size: int, rng: np.random.Generator) -> np.ndarray:
        """Genes centred just above zero so most assets start selected.

        A symmetric start around zero would leave half the universe deselected in
        every individual and make the early generations mostly about turning
        positions back on.
        """
        return rng.normal(loc=0.25, scale=self.init_scale, size=(size, problem.n))

    def step(self, X, fitness, problem, rng):     # pragma: no cover - overridden
        raise NotImplementedError

    def solve(self, problem, rng):
        budget = self.budget()
        X = self.initialise(problem, budget.pop_size, rng)
        fitness = score_encoded(X, problem)

        best_idx = int(np.argmax(fitness))
        best_x, best_f = X[best_idx].copy(), float(fitness[best_idx])
        history = [best_f]
        stale = 0

        for generation in range(budget.generations):
            X, fitness = self.step(X, fitness, problem, rng)
            gen_best = int(np.argmax(fitness))
            if fitness[gen_best] > best_f + budget.tol:
                best_f, best_x, stale = float(fitness[gen_best]), X[gen_best].copy(), 0
            else:
                stale += 1
            history.append(best_f)
            if stale >= budget.patience:
                log.debug("%s converged after %d generations", self.name, generation + 1)
                break

        return Allocation(
            decode(best_x, problem),
            self.name,
            0.0,
            history=history,
            diagnostics={
                "generations": len(history) - 1,
                "evaluations": len(history) * budget.pop_size,
                "converged_early": stale >= budget.patience,
            },
        )


@register
class GeneticAlgorithm(PopulationSearch):
    """Tournament selection, uniform crossover, Gaussian mutation, elitism."""

    name = "ga"

    def step(self, X, fitness, problem, rng):
        p = self.params
        size, n = X.shape
        elite = int(p.get("elite", 4))
        k = int(p.get("tournament_size", 3))
        p_cx = float(p.get("p_crossover", 0.9))
        p_mut = float(p.get("p_mutation", 0.15))
        sigma = float(p.get("sigma", 0.10))

        elite = min(elite, size - 2)
        order = np.argsort(-fitness)
        keep = X[order[:elite]] if elite > 0 else np.empty((0, n))
        n_children = size - elite

        # Tournament: k random contenders per slot, the fittest wins.
        contenders = rng.integers(0, size, size=(2 * n_children, k))
        winners = contenders[np.arange(2 * n_children), np.argmax(fitness[contenders], axis=1)]
        parents_a, parents_b = X[winners[:n_children]], X[winners[n_children:]]

        mask = rng.random((n_children, n)) < 0.5
        do_cx = rng.random((n_children, 1)) < p_cx
        offspring = np.where(mask & do_cx, parents_b, parents_a)

        mutate = rng.random((n_children, n)) < p_mut
        offspring = offspring + mutate * rng.normal(0.0, sigma, size=(n_children, n))

        X_new = np.vstack([keep, offspring])
        return X_new, score_encoded(X_new, problem)


@register
class DifferentialEvolution(PopulationSearch):
    """DE/rand/1/bin with binomial crossover and greedy one-to-one replacement."""

    name = "de"

    def step(self, X, fitness, problem, rng):
        size, n = X.shape
        F = float(self.params.get("F", 0.6))
        CR = float(self.params.get("CR", 0.9))

        idx = np.empty((size, 3), dtype=int)
        for i in range(size):
            pool = rng.choice(size - 1, size=3, replace=False)
            idx[i] = pool + (pool >= i)          # sample three distinct rows, never i
        mutant = X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]])

        cross = rng.random((size, n)) < CR
        forced = rng.integers(0, n, size=size)
        cross[np.arange(size), forced] = True    # at least one gene always comes from the mutant
        trial = np.where(cross, mutant, X)

        trial_fitness = score_encoded(trial, problem)
        better = trial_fitness > fitness
        return np.where(better[:, None], trial, X), np.where(better, trial_fitness, fitness)


@register
class ParticleSwarm(PopulationSearch):
    """Constricted PSO with the usual inertia and acceleration coefficients."""

    name = "pso"

    def solve(self, problem, rng):
        budget = self.budget()
        omega = float(self.params.get("omega", 0.72))
        c1 = float(self.params.get("c1", 1.5))
        c2 = float(self.params.get("c2", 1.5))
        vmax = float(self.params.get("vmax", 0.5))

        X = self.initialise(problem, budget.pop_size, rng)
        V = rng.normal(0.0, 0.1, size=X.shape)
        fitness = score_encoded(X, problem)

        pbest_x, pbest_f = X.copy(), fitness.copy()
        g = int(np.argmax(fitness))
        gbest_x, gbest_f = X[g].copy(), float(fitness[g])
        history, stale = [gbest_f], 0

        for _ in range(budget.generations):
            r1 = rng.random(X.shape)
            r2 = rng.random(X.shape)
            V = omega * V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest_x - X)
            V = np.clip(V, -vmax, vmax)
            X = X + V
            fitness = score_encoded(X, problem)

            improved = fitness > pbest_f
            pbest_x[improved], pbest_f[improved] = X[improved], fitness[improved]

            g = int(np.argmax(pbest_f))
            if pbest_f[g] > gbest_f + budget.tol:
                gbest_f, gbest_x, stale = float(pbest_f[g]), pbest_x[g].copy(), 0
            else:
                stale += 1
            history.append(gbest_f)
            if stale >= budget.patience:
                break

        return Allocation(
            decode(gbest_x, problem),
            self.name,
            0.0,
            history=history,
            diagnostics={
                "generations": len(history) - 1,
                "evaluations": len(history) * budget.pop_size,
            },
        )


@register
class Cmaes(PopulationSearch):
    """CMA-ES through the reference implementation.

    Included because it is the strongest general-purpose black-box optimiser in
    this class and because it adapts its own step size, which removes the "you
    tuned the genetic algorithm and not the baseline" objection.
    """

    name = "cmaes"

    def solve(self, problem, rng):
        import cma

        budget = self.budget()
        sigma0 = float(self.params.get("sigma0", 0.25))
        options = {
            "popsize": budget.pop_size,
            "maxiter": budget.generations,
            "tolfun": budget.tol,
            "seed": int(rng.integers(1, 2**31 - 1)),
            "verbose": -9,
        }
        es = cma.CMAEvolutionStrategy(np.full(problem.n, 0.25), sigma0, options)

        history = []
        while not es.stop():
            candidates = es.ask()
            values = score_encoded(np.asarray(candidates), problem)
            es.tell(candidates, (-values).tolist())          # cma minimises
            history.append(float(-es.result.fbest))

        best_x = np.asarray(es.result.xbest, dtype=float)
        return Allocation(
            decode(best_x, problem),
            self.name,
            0.0,
            history=history,
            diagnostics={
                "generations": len(history),
                "evaluations": int(es.result.evaluations),
                "stop": {k: str(v) for k, v in es.stop().items()},
            },
        )
