# Use NVIDIA's official CUDA Ubuntu image as a parent image
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    python3-pip \
    zip \
    unzip \
    nano \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && (rm -f /usr/bin/pip || true) \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# Clone the specific branch from the repository
#RUN git clone -b MSfinal https://github.com/kajc10/BRAIN2SPEECH.git
RUN git clone https://github.com/kajc10/BRAIN2SPEECH.git 

#COPY . /BRAIN2SPEECH

# Set the cloned directory as work directory
WORKDIR /BRAIN2SPEECH

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x ./download_dataset.sh

# Set up SSH server
RUN echo 'root:duck' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN mkdir /var/run/sshd

# Expose ports for TensorBoard and SSH
EXPOSE 6006 22

# By default, launch bash
CMD ["/bin/bash", "-c", "/usr/sbin/sshd && tail -f /dev/null"]



###############
# COMMANDS
###############
#docker build -t my-cuda-image .
#docker run --gpus all -it -p 6006:6006 -p 22:22 my-cuda-image
