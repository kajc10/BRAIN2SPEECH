# Use NVIDIA's official CUDA Ubuntu image as a parent image
FROM nvidia/cuda:11.0-base-ubuntu20.04

# Install necessary packages
RUN apt-get update && apt-get install -y git openssh-server

# Clone the repo
RUN git clone https://github.com/kajc10/BRAIN2SPEECH.git

# Set the cloned directory as work directory
WORKDIR /BRAIN2SPEECH

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Set up SSH server
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN mkdir /var/run/sshd

# Expose ports for TensorBoard and SSH
EXPOSE 6006 22

# By default, launch bash
CMD ["/bin/bash"]



###############
# COMMANDS
###############
#docker build -t my-cuda-image .
#docker run --gpus all -it -p 6006:6006 -p 22:22 my-cuda-image
