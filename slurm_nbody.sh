#!/bin/bash
#SBATCH --job-name=Nbody_sim     # Job name
#SBATCH --ntasks=4                  # Run on a 4 CPUs
#SBATCH --time=04:00:00             # Time limit hh:mm:ss
#SBATCH --output=job_output_%j.txt  # Standard output and error log

python main.py
