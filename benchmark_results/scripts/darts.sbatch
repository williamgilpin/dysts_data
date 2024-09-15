#!/bin/bash
# Job name:
#SBATCH --job-name=darts
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=gpu-a100
#
# Job progress file
#SBATCH --output=darts.out
#
# Error file:
#SBATCH --error=darts.err      
#
# Request one node:
#SBATCH --nodes=1
#
# number of tasks
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit: HH:MM:SS
#SBATCH --time=09:00:00
source /work/08962/wgilpin/ls6/miniforge3/bin/activate torch
mamba activate torch
python darts_benchmarks.py "$1"


