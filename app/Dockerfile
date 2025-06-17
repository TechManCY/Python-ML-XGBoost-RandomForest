FROM python:3.11-slim

WORKDIR /app

COPY dataCleaning.py .
COPY XGB.py .

RUN pip install psycopg2-binary pandas scikit-learn xgboost imblearn scipy numpy

CMD ["python", "XGB.py"]
