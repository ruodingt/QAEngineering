FROM tensorflow/tensorflow:1.12.0-gpu-py3

#FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter

ARG user
ARG password

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo "$user:$password" | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# docker build Docker --build-arg user=root --build-arg password=makefog -t aia:std-ssh
# docker run -it --rm --runtime=nvidia -p 6022:22 -p 6006:6006 -p 8888:8888 -d aia:std-ssh