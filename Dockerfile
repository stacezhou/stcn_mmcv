FROM stacezhou/base:cuda11.3
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda update conda
RUN conda create -y -n torch180 python=3.8
SHELL ["conda", "run", "-n", "torch180", "/bin/bash", "-c"]
# RUN conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
RUN conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# RUN conda install -y pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
RUN pip uninstall -y pillow  && pip install pillow-simd
RUN pip install progressbar2 gitpython gdown git+https://github.com/cheind/py-thin-plate-spline
RUN pip install tensorboard
RUN conda install -y gpustat
RUN echo 'conda activate torch180' >> /root/bashrc