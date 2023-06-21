FROM python:3.9
WORKDIR /app

ADD . /app
RUN apt-get update && apt-get install -y  gcc swig build-essential
RUN pip install --upgrade pip wheel==0.38.4 setuptools==65.5.1
RUN pip install -r requirements.txt
