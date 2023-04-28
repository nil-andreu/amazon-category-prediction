FROM python:3.8

LABEL version="0.1"
LABEL maintainer="Nil Andreu"

ENV PYTHONUTF8=1
ENV PYTHONPATH="."
ENV ENV_FILE_NAME="./api/main.py"

# Workdir & Volume
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .
VOLUME ./shared

EXPOSE 8080
CMD ["/bin/bash", "run.sh"]

