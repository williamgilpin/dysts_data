import time
import subprocess
from dysts.base import get_attractor_list

def submit_jobs():
    attractors = get_attractor_list()
    for attractor in attractors[110:]:
        #command = f"sbatch sweep_chronos.sbatch \"{attractor}\""
        command = f"sbatch sweep_darts.sbatch \"{attractor}\""
        subprocess.run(command, shell=True)
        time.sleep(0.65)

if __name__ == "__main__":
    submit_jobs()
