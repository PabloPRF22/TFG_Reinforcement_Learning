# Utilizar una imagen base de Python 3.7
FROM python:3.7

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos al directorio de trabajo
COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y python-dev libffi-dev libxml2-dev libxslt1-dev libyaml-dev swig
RUN apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev ffmpeg

# Instalar las bibliotecas requeridas
#RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copiar el resto de los archivos al directorio de trabajo
COPY . .