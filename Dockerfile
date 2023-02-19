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
RUN apt update 
COPY ./apt_packages.txt .
RUN xargs apt install --yes --no-install-recommends < apt_packages.txt \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Python packages
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir --requirement requirements.txt

# Clean up remaining things during install
RUN rm -rf /tmp/* /var/tmp/*

WORKDIR /app

CMD bash 