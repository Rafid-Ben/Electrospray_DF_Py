#!/bin/bash
#SBATCH --job-name=benchmark     # Job name
#SBATCH --ntasks=8                  # Run on a 8 CPUs
#SBATCH --time=01:00:00             # Time limit hh:mm:ss
#SBATCH --output=job_output_%j.txt  # Standard output and error log

python benchmark2.py
