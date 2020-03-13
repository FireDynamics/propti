# Ubuntu Image
FROM ubuntu:18.04

# download FDS installation script
ADD https://github.com/firemodels/fds/releases/download/FDS6.7.4/FDS6.7.4_SMV6.7.14_lnx.sh /root/

# install FDS
RUN cd /root/ && bash FDS6.7.4_SMV6.7.14_lnx.sh y
RUN . /root/FDS/FDS6/bin/FDS6VARS.sh

# set environment for FDS
ENV LD_LIBRARY_PATH=/root/FDS/FDS6/bin/INTEL/lib/
ENV PATH=/root/FDS/FDS6/bin:$PATH

# setup software repository
RUN apt-get update

# install python
RUN apt-get install -y python3-pip

# install python packages
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install spotpy

# install git, ssh, compilers
RUN apt-get install -y git ssh gcc gfortran

# install MPI
RUN apt-get install -y libmpich-dev libmpich12 mpich
# RUN apt-get install -y libopenmpi-dev openmpi-bin

# clean up software installation
RUN apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install MPI for Python
RUN pip3 install mpi4py

# install propti
RUN git clone https://github.com/FireDynamics/propti.git /root/propti/

RUN echo "alias propti_prepare='python3 /root/propti/propti_prepare.py'" >> /root/.bashrc
RUN echo "alias propti_run='python3 /root/propti/propti_run.py'" >> /root/.bashrc
RUN echo "source /root/FDS/FDS6/bin/FDS6VARS.sh" >> /root/.bashrc

WORKDIR /workdir
