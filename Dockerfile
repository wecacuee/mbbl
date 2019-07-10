FROM tensorflow/tensorflow:latest-gpu-py3

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN pip install tox pytest
