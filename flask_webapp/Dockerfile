FROM alpine
MAINTAINER daniele.gadler@siag.it
FROM python:3.7

ENV HTTP_PROXY=http://10.240.150.134:8020
ENV HTTPS_PROXY=http://10.240.150.134:8020

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["flask", "run", "--host=0.0.0.0"]

#Run with "docker run -p 5000:5000 <image_name>"