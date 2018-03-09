#!/bin/bash ­x
# submit a chain of jobs with dependency

# number of jobs to submit
NO_OF_JOBS=XX

# define jobscript
JOB_SCRIPT=fds-chain-part-modified.job

echo "sbatch ${JOB_SCRIPT}"
JOBID=$(sbatch ${JOB_SCRIPT} 2>&1 | awk '{print $(NF)}')

# Launch the next job, after the previous one has been finished successfully.
I=0
while [ ${I} -le ${NO_OF_JOBS} ]; do
  echo "sbatch -d afterok:${JOBID} ${JOB_SCRIPT}"
  JOBID=$(sbatch -d afterok:${JOBID} ${JOB_SCRIPT} 2>&1 | awk '{print $(NF)}')
  let I=${I}+1
done
