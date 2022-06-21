FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && apt-get update && apt-get install -y --no-install-recommends git

WORKDIR /workspace

# install fairseq
RUN git clone https://github.com/pytorch/fairseq \
  && git clone https://github.com/NVIDIA/apex \
  && cd fairseq \
  && pip install -i https://pypi.douban.com/simple --no-cache-dir --editable ./ \
  && cd ../apex \
  && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./ \
  && pip install -i https://pypi.douban.com/simple --no-cache-dir pyarrow

# install sentencepiece
ENV PATH /WORKSPACE/sentencepiece/build/src:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
  cmake build-essential pkg-config libgoogle-perftools-dev \
  && git clone https://github.com/google/sentencepiece.git \
  && cd sentencepiece \
  && mkdir build && cd build \
  && cmake .. && make -j $(nproc) \
  && make install && ldconfig -v \
  && pip install -i https://pypi.douban.com/simple --no-cache-dir sentencepiece

# install eflomal
ENV EFLOMAL_PATH /workspace/eflomal
RUN git clone https://github.com/robertostling/eflomal \
  && cd eflomal \
  && make \
  && make install \
  && python3 setup.py install

# insntall opusfilter
RUN git clone https://github.com/BrightXiaoHan/OpusFilter.git \
  && cd OpusFilter \
  && pip install -i https://pypi.douban.com/simple --no-cache-dir --editable .[all]

# install translate server
RUN git clone https://github.com/BrightXiaoHan/lairmtdeploy.git \
  && cd lairmtdeploy \
  && pip install -i https://pypi.douban.com/simple --no-cache-dir --editable .[all]


# other env
ENV DATA_DIR /root/data
ENV TRAIN_DIR /root/train

ADD . /workspace/lanmttrainer
WORKDIR /workspace/lanmttrainer

ENTRYPOINT ["/bin/bash"]
