FROM python:3.7.3-slim

RUN apt-get update && apt-get install -y locales

RUN sed -i -e \
    's/# ru_RU.UTF-8 UTF-8/ru_RU.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen

ENV LANG ru_RU.UTF-8
ENV LANGUAGE ru_RU:ru
ENV LC_LANG ru_RU.UTF-8
ENV LC_ALL ru_RU.UTF-8

RUN mkdir /ImageCaptioning
WORKDIR /ImageCaptioning
COPY . /ImageCaptioning
RUN pip install -r requirements.txt