#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=2
#SBATCH --time=0-02
#SBATCH -p batch-short


cd $SLURM_SUBMIT_DIR

pwd

printenv &> user.env-$SLURM_JOB_ID

which python3

which fds653_serial

module load mpi/mpich/3.2/gcc-6.3.1

export PATH=$PATH:/hpcscratch/user/username/path/to/FDS
mpirun python3 /hpcscratch/user/username/path/to/propti/propti/propti_run.py . &> log.spotpy_mpi


