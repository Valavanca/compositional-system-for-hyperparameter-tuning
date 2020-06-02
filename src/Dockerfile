FROM python:3.8-slim

WORKDIR /usr/src/app

# System packages 
RUN apt-get update && apt-get install -y curl

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda


COPY requirements.txt ./
RUN conda config --add channels conda-forge && \
    conda install pygmo && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# RUN [ "python", "./benchmark.py" ]
CMD [ "bash" ]