#!/bin/bash
#SBATCH -J citebench
#SBATCH -A r01682
#SBATCH -p h100-debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /N/project/mrust01/citation_agents_system/benchmark/data/slurm-%j.log
# Generic GPU job for the benchmark scripts. Usage: sbatch run_job.slurm "<shell command>"
source /etc/profile.d/modules.sh
module load python/gpu/3.12.5
cd /N/project/mrust01/citation_agents_system/benchmark
set -x
eval "$1"
