FROM openface:base

COPY ./app app

WORKDIR app

RUN pip install -r requirements.txt && \
    pip install pygem-2.0.0.tar.gz

CMD ["streamlit", "run", "app.py", "--server.port", "80"]