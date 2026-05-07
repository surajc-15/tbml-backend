import json
import time
import random
import uuid
from datetime import datetime
from kafka import KafkaProducer

# --- CONFIGURATION ---
KAFKA_TOPIC = 'incoming_swift_messages'

COMMODITY_PRICES = {
    'Textiles': 50.0, 'Electronics': 450.0, 'Scrap Metal': 120.0, 
    'Luxury Goods': 1200.0, 'Grain': 30.0
}

# A pool of known accounts to simulate normal network traffic
KNOWN_ACCOUNTS = [f"ACC_{i:06d}" for i in range(100)] 

print("⚙️ Initializing Advanced Stateful Kafka Producer...")
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("🚀 Starting Advanced Synthetic Live Stream...\n")
print("-" * 75)

try:
    while True:
        # 1. Base Transaction Setup
        msg_id = f"SW_LIVE_{str(uuid.uuid4())[:8].upper()}"
        doc_id = f"DOC_LIVE_{str(uuid.uuid4())[:8].upper()}"
        
        # 2. Decide the nature of the transaction (Now with 3 Classes!)
        rand_roll = random.random()
        if rand_roll < 0.05:
            tx_class = 2  # 5% chance of Full Fraud
        elif rand_roll < 0.15:
            tx_class = 1  # 10% chance of Watchlist (Suspicious, but not blatant)
        else:
            tx_class = 0  # 85% chance of Clean

        is_cold_start = random.random() < 0.20  # 20% chance of a brand new account appearing

        if is_cold_start:
            sender = f"ACC_UNKNOWN_{random.randint(1000, 9999)}"
            receiver = random.choice(KNOWN_ACCOUNTS)
        else:
            sender = random.choice(KNOWN_ACCOUNTS)
            receiver = random.choice([acc for acc in KNOWN_ACCOUNTS if acc != sender])

        # 3. Trade Specifics (Introducing Real-World Noise)
        commodity = random.choice(list(COMMODITY_PRICES.keys()))
        market_avg = COMMODITY_PRICES[commodity]
        qty = random.randint(10, 2000)
        declared_weight_kg = qty * 2.2

        if tx_class == 2:
            # CLASS 2 (FRAUD): Extreme anomalies
            unit_price = market_avg * random.uniform(3.0, 5.0)
            actual_weight_kg = declared_weight_kg * random.uniform(0.75, 0.85)
            # Noise: 10% of Cartels remember to turn their AIS ON to look normal
            ais_status = 'ON' if random.random() < 0.10 else random.choice(['OFF', 'DARK'])
            port_status = 'UNVERIFIED'
            pattern = random.choice(['FAN-IN', 'CYCLE', 'STACK'])
            
        elif tx_class == 1:
            # CLASS 1 (WATCHLIST): Mild anomalies that warrant a human check
            unit_price = market_avg * random.uniform(1.5, 2.5) # A little high, but maybe just inflation?
            actual_weight_kg = declared_weight_kg * random.uniform(0.85, 0.95) # Slight missing cargo
            # Noise: AIS is usually ON, but sometimes cuts out due to weather/tech issues
            ais_status = 'OFF' if random.random() < 0.30 else 'ON'
            port_status = random.choice(['VERIFIED', 'UNVERIFIED'])
            pattern = 'SUSPICIOUS_TRANSFER'
            
        else:
            # CLASS 0 (CLEAN): Normal business
            unit_price = market_avg * random.uniform(0.95, 1.05)
            actual_weight_kg = declared_weight_kg * random.uniform(0.98, 1.0)
            # Noise: Even legal ships have broken transponders sometimes (2% chance)
            ais_status = 'OFF' if random.random() < 0.02 else 'ON'
            port_status = 'VERIFIED'
            pattern = 'LEGIT'

        amount = round(qty * unit_price, 2)
        price_deviation = round(unit_price - market_avg, 2)
        weight_gap_score = round(declared_weight_kg - actual_weight_kg, 2)

        # 4. Construct the Enriched Payload
        payload = {
            "swift": {
                "msg_id": msg_id,
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": amount,
                "currency": "USD",
                "timestamp": datetime.now().isoformat(),
                "pattern_type": pattern 
            },
            "trade": {
                "doc_id": doc_id,
                "commodity": commodity,
                "qty": qty,
                "unit_price": round(unit_price, 2),
                "market_avg": market_avg,
                "price_deviation": price_deviation,
                "declared_weight_kg": round(declared_weight_kg, 2),
                "actual_weight_kg": round(actual_weight_kg, 2),
                "weight_gap_score": weight_gap_score,
                "ais_status": ais_status,
                "port_log_status": port_status
            }
        }
        
        # 5. Send to Kafka
        producer.send(KAFKA_TOPIC, value=payload)
        
        # 6. Clean Console Logging (Updated for 3 classes)
        if tx_class == 2:
            status_emoji = "🚨 FRAUD "
        elif tx_class == 1:
            status_emoji = "⚠️ WATCH "
        else:
            status_emoji = "✅ LEGIT "
            
        print(f"[{time.strftime('%H:%M:%S')}] {status_emoji} | {msg_id} | ${amount:,.2f} | {commodity} (AIS: {ais_status})")
        if tx_class in [1, 2]:
            print(f"    ↳ Anomaly Details: Dev: {price_deviation:.2f} | Weight Gap: {weight_gap_score:.2f}kg | Pattern: {pattern}")
        
        # Sleep to simulate network delay
        time.sleep(random.uniform(1.5, 3.5))

except KeyboardInterrupt:
    print("\n🛑 Stopping advanced synthetic stream.")
finally:
    producer.close()