import json
import logging
import os
from kafka import KafkaProducer
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# --- CONFIGURATION ---
KAFKA_TOPIC = 'fraud_alerts'
KAFKA_SERVER = os.getenv("KAFKA_BROKER", "localhost:19092")
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PW = "password"  # Ensure this matches your instance

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
            # Initialize Neo4j Driver
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PW))
            logger.info("✅ Connected to Neo4j and Kafka successfully.")
        except Exception as e:
            logger.error(f"❌ Initialization Failed: {e}")
            raise

    def close(self):
        self.driver.close()
        self.producer.close()

    def stream_phantom_data(self):
        """
        Extracts high-risk Singapore-UK Mesh data and streams to Kafka.
        """
        logger.info("🔍 Running Graph Forensics Query...")
        
        # This query targets the specific 'Smartphone/Garment' pattern we found
        query = """
        MATCH (e:Entity)-[:SENDS]->(t:Transaction)-[:HAS_DOCUMENT]->(d:Document)
        WHERE d.ghost = 1 
           OR d.weight_gap >= 1.0 
           OR e.device_hash IN ['DEV_HASH_011', 'DEV_HASH_003', 'DEV_HASH_042']
        RETURN e.entity_id AS sender, 
               e.device_hash AS device,
               e.country AS origin,
               t.msg_id AS txn_id, 
               t.amount AS amount, 
               t.timestamp AS ts,
               d.commodity AS goods, 
               d.weight_gap AS gap,
               d.price_dev AS price_inf
        """
        
        try:
            count = 0
            with self.driver.session() as session:
                results = session.run(query)
                for record in results:
                    # Construct the forensic payload
                    payload = {
                        "alert_type": "HIGH_RISK_TBML",
                        "sender_id": record["sender"],
                        "device_fingerprint": record["device"],
                        "transaction": {
                            "id": record["txn_id"],
                            "amount": record["amount"],
                            "currency": "USD",
                            "timestamp": record["ts"]
                        },
                        "fraud_indicators": {
                            "commodity": record["goods"],
                            "weight_gap": record["gap"],
                            "price_inflation": record["price_inf"],
                            "is_ghost": True if record["gap"] >= 1.0 else False
                        },
                        "origin_country": record["origin"],
                        "source_system": "Neo4j_Fraud_Graph"
                    }
                    
                    # Send to Kafka with a callback for confirmation
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