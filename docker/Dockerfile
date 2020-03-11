# Ubuntu Image
FROM ubuntu:18.04

# download FDS installation script
ADD https://github.com/firemodels/fds/releases/download/FDS6.7.4/FDS6.7.4_SMV6.7.14_lnx.sh /root/

# install FDS
RUN cd /root/ && bash FDS6.7.4_SMV6.7.14_lnx.sh y

# set environment for FDS
ENV LD_LIBRARY_PATH=/root/FDS/FDS6/bin/INTEL/lib/
ENV PATH=/root/FDS/FDS6/bin:$PATH

# setup software repository
RUN apt-get update

# install python
RUN apt-get install -y python3-pip

# install python packages
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install spotpy

# install git
RUN apt-get install -y git

# install propti
RUN git clone https://github.com/FireDynamics/propti.git /root/propti/

WORKDIR /workdir
