#!/bin/bash
#SBATCH --job-name=scen_pub
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Last link in the chain: push the tables and figures back to the remote.
#
# Needs the compute nodes to reach github.com. If they cannot, drop this job from
# submit_all.sh and run `bash scripts/publish.sh` on the login node once the
# aggregation has finished.

source "$(dirname "$0")/env.sh"

bash scripts/publish.sh "results from job chain ${SLURM_JOB_ID}"
