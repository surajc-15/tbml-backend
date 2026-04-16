# Clean trigger.py
import json
import os
from kafka import KafkaProducer
from neo4j import GraphDatabase

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")

driver = GraphDatabase.driver("bolt://localhost:4000", auth=("", ""))
with driver.session() as session:
    # Let's specifically find a transaction we KNOW is dirty based on your rules
    # We are hunting for a massive coordinated fraud event
    query = """
    MATCH (s:Account)-[:SENDS]->(t:Transaction)-[:TO]->(r:Account)
    MATCH (t)-[:HAS_DOC]->(d:Document)
    WHERE d.price_deviation > 3.0 
      AND d.weight_gap_score > 0.15 
      AND d.ais_status = 'DARK'
      AND s.dorm_days > 180
    RETURN t.id AS msg_id, t.amount AS amount LIMIT 1
    """
    result = session.run(query).single()
    real_msg_id = result["msg_id"]
    real_amount = result["amount"]
driver.close()

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

tx = {"msg_id": real_msg_id, "amount": real_amount, "currency": "USD"}
producer.send('incoming_swift_messages', tx)
producer.flush()
print(f"✅ Fired actual historical transaction: {real_msg_id}")