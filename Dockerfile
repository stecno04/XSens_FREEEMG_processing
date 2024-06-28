FROM python:3.11.9-bullseye
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app
# ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1
# RUN apk add --update --no-cache --virtual .tmp gcc libc-dev linux-headers
# RUN apk add --no-cache jpeg-dev zlib-dev mariadb-dev libffi-dev openblas-dev libgfortran lapack-dev build-base openssl-dev
# RUN apk add --no-cache hdf5 hdf5-dev
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r requirements.txt
COPY for_docker/ /app/for_docker
COPY analysis_functions/ /app/analysis_functions
CMD [ "python3", "liveProcessing/processing.py"]

# always add --add-host mlflow_serv:10.250.4.35 when running
