import json
import os
from kafka import KafkaConsumer

# Configuration
TOPIC = "fraud_alerts"
BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")

def calculate_risk(tx):
    """Simple Logic-based GNN proxy for TBML detection"""
    score = 0.0
    # Red Flag 1: The 'Ghost Trade' (Weight Gap is 1.0)
    if tx.get('logistics_data', {}).get('weight_gap') == 1.0:
        score += 0.5
    
    # Red Flag 2: High Value for low-value commodities (Jute/Garments)
    amount = tx.get('amount', 0)
    commodity = tx.get('logistics_data', {}).get('commodity', '')
    if amount > 1000000:
        score += 0.4
    elif amount > 500000:
        score += 0.2
        
    return round(min(score, 1.0), 2)

def start_consumer():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BROKER],
        auto_offset_reset='earliest', # Read the 5,500 messages already there
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print(f"🚀 Neural Pipeline Active. Monitoring {TOPIC}...")
    
    for message in consumer:
        tx = message.value
        risk_score = calculate_risk(tx)
        
        if risk_score >= 0.7:
            print(f"🚨 [HIGH RISK] ID: {tx['transaction_id']} | Sender: {tx['sender_id']} | Risk: {risk_score}")
            print(f"   Details: {tx['amount']:.2f} USD for {tx['logistics_data']['commodity']}")
        elif risk_score >= 0.4:
            print(f"⚠️  [SUSPICIOUS] ID: {tx['transaction_id']} | Risk: {risk_score}")

if __name__ == "__main__":
    start_consumer()