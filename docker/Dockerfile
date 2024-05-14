# FDS version (default: latest), e.g. 6.9.1
ARG FDS_VERSION=latest

# FDS base image
FROM ghcr.io/openbcl/fds:${FDS_VERSION}

# install app dependencies
RUN apt-get update && apt-get install -y libmpich-dev python3-pip python3-venv

# create python venv for propti
RUN python3 -m venv /opt/venv/propti

# append python venv for propti to path
ENV PATH="/opt/venv/propti/bin:$PATH"

# install python dependencies
RUN pip install matplotlib mpi4py numpy pandas scipy spotpy

# copy propti
COPY ./ /root/propti/

# create wrapper script, fix permissions and create symlinks
RUN echo '/usr/bin/env bash -l -c "$*"' > /root/propti/wrap.sh && \
    chmod +x /root/propti/wrap.sh && \
    chmod +x /root/propti/*.py && \
    rm -rf /root/propti/docker && \
    ln -s /root/propti/wrap.sh /bin/wrap && \
    ln -s /root/propti/propti_analyse.py /bin/propti_analyse && \
    ln -s /root/propti/propti_prepare.py /bin/propti_prepare && \
    ln -s /root/propti/propti_run.py /bin/propti_run && \
    ln -s /root/propti/propti_sense.py /bin/propti_sense