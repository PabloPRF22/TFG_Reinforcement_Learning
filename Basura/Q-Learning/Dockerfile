# Utilizar una imagen base de Python 3.7
FROM python:3.8

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos al directorio de trabajo
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y gcc cmake python-dev libffi-dev libxml2-dev libxslt1-dev libyaml-dev swig

# Instalar las bibliotecas requeridas
RUN pip install -r requirements.txt

# Copiar el resto de los archivos al directorio de trabajo
COPY . .

# Exponer el puerto para la aplicación (si es necesario)
# EXPOSE 80

# Ejecutar el script de Python cuando se inicie el contenedor
