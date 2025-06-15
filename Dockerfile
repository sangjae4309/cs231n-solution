FROM pytorch/pytorch
LABEL creator="sangjae4309@gmail.com"
LABEL version="1.0.0"

USER root
WORKDIR /root

RUN apt-get update 
RUN apt-get install build-essential -y
RUN apt-get install wget
RUN apt-get install unzip
RUN apt-get install git
RUN pip install matplotlib
RUN pip install h5py
RUN pip install imageio
RUN pip install scipy
RUN pip install ipywidgets
RUN pip install pandas

RUN git clone https://github.com/sangjae4309/cs231n-solution.git /root/
