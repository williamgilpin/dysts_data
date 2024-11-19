#!/bin/bash
# Job name:
#SBATCH --job-name=incontext_chronos
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=normal
#
# Job progress file
#SBATCH --output=incontext.out
#
# Error file:
#SBATCH --error=incontext.err      
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
#SBATCH --time=12:00:00
#
source /work/08962/wgilpin/ls6/miniforge3/bin/activate
mamba activate dysts_benchmarks
python chronos_incontext.py "$1" "${2:-30}"

