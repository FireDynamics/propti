#!/usr/bin/sh

env -i bash -l -c "
echo 'running FDS'

echo 'loading modules'
module use -a ~arnold1/modules_fire/
module load FDS/6.7.4-IntelComp2019.0_ParaStationMPI_5.2.1

which fds

if [ \"$#\" -eq 0 ]; then
    echo | OMP_NUM_THREADS=1 fds
else
    OMP_NUM_THREADS=1 fds $1
    wct=\`grep 'Total Elapsed Wall Clock Time' *.out | cut -d: -f2\`
    hostname=`hostname`
    dir=${PWD##*}
    date=\`date\`
    echo \"`date`; `hostname`; \`grep 'Total Elapsed Wall Clock Time' *.out | cut -d: -f2\`; `pwd` \"  > wct.csv
fi
"
