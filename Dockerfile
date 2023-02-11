FROM ubuntu:20.04

ENV TZ=Etc/UTC
ARG DEBIAN_FRONTEND=noninteractive
ARG ROOT_PWD=root

USER root
# Set root password
RUN echo 'root:${ROOT_PWD}' | chpasswd

# Install things here
WORKDIR /tmp

# Install 
RUN apt update \
    && apt install -y --no-install-recommends \
    python3 python3-opencv

# Python packages
RUN pip3 install ezdxf

# Clean up
RUN apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up env
# RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

WORKDIR /app

CMD bash 