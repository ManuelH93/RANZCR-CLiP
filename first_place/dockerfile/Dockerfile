FROM nvidia/cuda:10.2-base
CMD nvidia-smi

FROM pytorch/pytorch:latest


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install opencv-python
RUN apt-get -y install git
RUN python3 -m pip install matplotlib
RUN python3 -m pip install pandas
RUN python3 -m pip install torchsummary
RUN python3 -m pip install tiffile
RUN python3 -m pip install albumentations
RUN python3 -m pip install timm
RUN python3 -m pip install seaborn
RUN python3 -m pip --no-cache-dir install --upgrade awscli