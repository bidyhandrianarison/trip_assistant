FROM python:3.13.11-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY trip_rag.py travel_data.json .

ENTRYPOINT ["python", "trip_rag.py"]