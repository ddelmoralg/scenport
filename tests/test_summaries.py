"""Per-run summaries must not mix a sensitivity variant into its base generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import aggregate  # noqa: E402


def _write(directory: Path, run: str, allocator: str, sharpe: float) -> None:
    rows = [{"forecaster": "garch", "allocator": allocator, "seed": s,
             "label": "garch+%s+s%d" % (allocator, s), "sharpe": sharpe} for s in range(5)]
    pd.DataFrame(rows).to_parquet(directory / f"{run}__{allocator}.parquet")


def test_an_ablation_is_not_averaged_into_its_base_run(tmp_path):
    summary = tmp_path / "summary"
    summary.mkdir()
    # Both files say forecaster == "garch" and share seeds 0 to 4, which is how
    # the ablation job writes them.
    _write(summary, "garch", "ga", sharpe=0.5)
    _write(summary, "garch_card05", "ga", sharpe=0.9)

    base = aggregate.load_summaries(tmp_path)
    assert set(base["forecaster"]) == {"garch"}
    assert base["sharpe"].mean() == 0.5

    every = aggregate.load_summaries(tmp_path, include_ablations=True)
    assert set(every["forecaster"]) == {"garch", "garch_card05"}
    assert len(every) == 10
