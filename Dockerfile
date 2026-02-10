FROM python:3.10-slim-buster
# my working directory in docker
WORKDIR /app
# the directory in app folder in dockers will copy the all the contents of the project 
COPY . /app

# update evething (all the files) and download the awscli 
RUN apt update -y && install awscli -y 

# update this and install requirements folder
RUN apt-get upadte && pip install -r requirements.txt
CMD ["pyhton3","app.py"]