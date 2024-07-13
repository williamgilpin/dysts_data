#!/bin/bash
# Job name:
#SBATCH --job-name=chronos
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=normal
#
# Job progress file
#SBATCH --output=chronos.out
#
# Error file:
#SBATCH --error=chronos.err      
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
#SBATCH --time=08:30:00
#
mamba activate dysts_benchmarks
python chronos_benchmarks.py "$1"

