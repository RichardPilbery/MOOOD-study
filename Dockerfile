FROM python:3.9-slim

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update
RUN apt-get -y install git
RUN git clone https://github.com/RichardPilbery/MOOOD-study.git

RUN apt-get -y install nano
RUN apt-get -y install gcc python3-dev
RUN apt-get -y install tk

RUN cp -a MOOOD-study/. . && rm -r MOOOD-study
RUN mkdir -p -v data

RUN pip install -r ./requirements.txt

CMD gunicorn -b 0.0.0.0:80 index:server
