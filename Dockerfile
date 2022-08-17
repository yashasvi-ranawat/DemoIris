    FROM python:3.10
    LABEL prod "prediction"
    EXPOSE 8000
    ENV PROJECT_DIR /usr/local/src/webapp
    COPY app ${PROJECT_DIR}  
    WORKDIR ${PROJECT_DIR}
    RUN ["pip", "install", "pipenv"]
    RUN ["python", "-m", "pipenv", "install", "--deploy"]
    ENTRYPOINT  ["python", "-m", "pipenv", "run", "uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]  
