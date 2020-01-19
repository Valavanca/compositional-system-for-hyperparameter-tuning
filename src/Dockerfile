FROM python:alpine3.7

COPY . /app
WORKDIR /app

# Project dependencies
RUN pip3 install --no-cache-dir -q -r requirements.txt

RUN python ./benchmark.py
CMD [ "bash" ]