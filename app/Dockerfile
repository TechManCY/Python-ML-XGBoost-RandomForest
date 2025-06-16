FROM python:3.11-slim

WORKDIR /app

COPY dataPreProcessing.py .

RUN pip install psycopg2-binary pandas

CMD ["python", "dataPreProcessing.py"]
