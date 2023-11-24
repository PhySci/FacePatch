FROM python:3.9

COPY ./app/requirements.txt /app/requirements.txt
COPY ./app/pygem-2.0.0.tar.gz /app/pygem.tar.gz

WORKDIR /app
RUN pip install -r requirements.txt  && \
    pip install pygem.tar.gz