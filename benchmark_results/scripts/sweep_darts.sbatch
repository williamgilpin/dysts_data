#!/bin/bash
# Job name:
#SBATCH --job-name=sweep_darts
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=gpu-a100-small
#
# Job progress file
#SBATCH --output=sweep_darts.out
#
# Error file:
#SBATCH --error=sweep_darts.err      
#
# Request one node:
#SBATCH --nodes=1
# 
# Use all available memory
#SBATCH --mem=0  
#
# number of tasks
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit: HH:MM:SS 12 hr
#SBATCH --time=04:00:00
#
source /work/08962/wgilpin/ls6/miniforge3/bin/activate
mamba activate dysts_benchmarks
python sweep_darts.py "$1"

