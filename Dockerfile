FROM python:3.9-slim
WORKDIR /app
RUN pip install pandas neo4j pyarrow
COPY neo4j_ingestion.py .
CMD ["python", "neo4j_ingestion.py"]