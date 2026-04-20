import json
import logging
import os
from kafka import KafkaProducer
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# --- CONFIGURATION ---
KAFKA_TOPIC = 'incoming_swift_messages'
KAFKA_SERVER = os.getenv("KAFKA_BROKER", "localhost:19092")
MEMGRAPH_URI = "bolt://localhost:4000"
MEMGRAPH_USER = ""
MEMGRAPH_PW = ""

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudIntelligenceProducer:
    def __init__(self):
        try:
            # Initialize Kafka Producer
            self.producer = KafkaProducer(
                bootstrap_servers=[KAFKA_SERVER],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                acks='all',  # Ensure delivery
                retries=3
            )
            # Initialize Memgraph Driver
            self.driver = GraphDatabase.driver(MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PW))
            logger.info("✅ Connected to Memgraph and Kafka successfully.")
        except Exception as e:
            logger.error(f"❌ Initialization Failed: {e}")
            raise

    def close(self):
        self.driver.close()
        self.producer.close()

    def stream_phantom_data(self):
        """
        Extracts real SWIFT transactions from Memgraph and simulates a real-time stream.
        """
        logger.info("🔍 Fetching live transactions from Memgraph...")
        
        query = """
        MATCH (t:Transaction)
        RETURN t.id AS txn_id, t.amount AS amount
        LIMIT 50
        """
        
        try:
            count = 0
            with self.driver.session() as session:
                results = session.run(query)
                for record in results:
                    # Construct the pipeline-ready payload
                    payload = {
                        "msg_id": record["txn_id"],
                        "amount": record["amount"],
                        "currency": "USD"
                    }
                    
                    # Send to Kafka so inference.py catches it
                    self.producer.send(KAFKA_TOPIC, value=payload).add_callback(self.on_success).add_errback(self.on_error)
                    count += 1
            
            self.producer.flush()
            logger.info(f"✅ Batch Stream Complete. {count} alerts pushed to Kafka.")

        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")

    def on_success(self, record_metadata):
        pass # Optional: log successful offsets

    def on_error(self, exc):
        logger.error(f"❌ Kafka Delivery Error: {exc}")

if __name__ == "__main__":
    streamer = FraudIntelligenceProducer()
    try:
        streamer.stream_phantom_data()
    finally:
        streamer.close()