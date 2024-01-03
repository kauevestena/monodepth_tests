FROM pytorch/pytorch:latest

RUN apt update

RUN apt install -y git

RUN conda create -n iebins python=3.8

RUN conda activate iebins

RUN conda install pytorch=1.10.0 torchvision cudatoolkit=11.1

RUN pip install matplotlib, tqdm, tensorboardX, timm, mmcv, open3d

RUN git clone https://github.com/ShuweiShao/IEBins.git

RUN git clone https://github.com/kauevestena/monodepth_tests.git
