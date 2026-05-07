import json
import torch
import numpy as np
from kafka import KafkaConsumer
from neo4j import GraphDatabase
from model import TBML_DetectionModel 

# --- CONFIGURATION ---
MEMGRAPH_URI = "bolt://localhost:4000"
MEMGRAPH_USER = ""
MEMGRAPH_PASS = ""
KAFKA_TOPIC = 'incoming_swift_messages'

# ==========================================
# PHASE 1: SYSTEM PRE-FLIGHT
# ==========================================
print("⚙️ Booting Real-Time AML Inference Engine...")

model = TBML_DetectionModel(kyc_dim=3, swift_dim=1, trade_dim=3, hidden_dim=64, lstm_hidden=64, classes=3)

try:
    print("📂 Loading weights from global_model.npz...")
    npz_data = np.load('global_model.npz')
    
    state_dict = model.state_dict()
    loaded_weights = [npz_data[f"arr_{i}"] for i in range(len(state_dict))]
    
    for i, key in enumerate(state_dict.keys()):
        state_dict[key] = torch.tensor(loaded_weights[i])
        
    model.load_state_dict(state_dict)
    model.eval() 
    print("✅ Neural Network Weights Successfully Injected.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load weights. Error: {e}")
    exit()

db_driver = GraphDatabase.driver(MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASS))
print("✅ Connected to Memgraph Database.")

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)
print("✅ Kafka Consumer Subscribed. Listening for live transactions...\n")
print("=" * 75)


# ==========================================
# PHASE 2: DATABASE INGESTION
# ==========================================
def ingest_live_transaction(payload):
    swift = payload['swift']
    trade = payload.get('trade', {})

    query = """
    MERGE (sender:Account {id: $sender_id})
    MERGE (receiver:Account {id: $receiver_id})
    MERGE (t:Transaction {id: $msg_id})
    SET t.amount = $amount, t.currency = $currency
    MERGE (sender)-[:SENDS]->(t)-[:TO]->(receiver)
    WITH t
    WHERE $doc_id IS NOT NULL
    MERGE (d:Document {id: $doc_id})
    SET d.commodity = $commodity, 
        d.price_deviation = $price_dev, 
        d.weight_gap_score = $weight_gap, 
        d.ais_status = $ais
    MERGE (t)-[:HAS_DOC]->(d)
    """
    params = {
        "sender_id": swift['sender_id'], "receiver_id": swift['receiver_id'],
        "msg_id": swift['msg_id'], "amount": swift['amount'], "currency": swift['currency'],
        "doc_id": trade.get('doc_id'), "commodity": trade.get('commodity'),
        "price_dev": trade.get('price_deviation'), "weight_gap": trade.get('weight_gap_score'),
        "ais": trade.get('ais_status')
    }
    with db_driver.session() as session:
        session.run(query, **params)


# ==========================================
# PHASE 3: EXACT TENSOR REPLICATION
# ==========================================
def get_inference_tensors(tx_id, swift_payload, trade_payload):
    query = """
    MATCH (sender:Account)-[:SENDS]->(t:Transaction {id: $tx_id})-[:TO]->(receiver:Account)
    RETURN sender, receiver
    """
    with db_driver.session() as session:
        result = session.run(query, tx_id=tx_id).single()
        
    sender, receiver = result['sender'], result['receiver']

    kyc_x = torch.tensor([
        [float(sender.get("is_shell", 0)), float(sender.get("dorm_days", 0)), float(sender.get("device_entropy", 0))],
        [float(receiver.get("is_shell", 0)), float(receiver.get("dorm_days", 0)), float(receiver.get("device_entropy", 0))]
    ], dtype=torch.float32)

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    raw_amount = float(swift_payload.get('amount', 0))
    swift_edge_attr = torch.tensor([[raw_amount]], dtype=torch.float32)

    seq_data = swift_edge_attr.unsqueeze(0)

    price_dev = float(trade_payload.get('price_deviation', 0.0))
    weight_gap = float(trade_payload.get('weight_gap_score', 0.0))
    ais_dark = 1.0 if trade_payload.get('ais_status') in ['DARK', 'OFF'] else 0.0
    
    trade_features = torch.tensor([[price_dev, weight_gap, ais_dark]], dtype=torch.float32)

    return kyc_x, edge_index, swift_edge_attr, seq_data, trade_features


# ==========================================
# PHASE 4: THE REAL-TIME PIPELINE LOOP
# ==========================================
def process_live_transaction(payload):
    swift = payload['swift']
    trade = payload.get('trade', {}) 
    tx_id = swift['msg_id']
    
    # --- RATIO FIX INTERCEPTION ---
    # We catch the raw data from Kafka and convert it to the strict 1.0 - 5.0 ratios the AI expects
    if trade:
        unit_price = float(trade.get('unit_price', 1.0))
        market_avg = float(trade.get('market_avg', 1.0))
        declared_weight = float(trade.get('declared_weight_kg', 1.0))
        weight_gap_raw = float(trade.get('weight_gap_score', 0.0))

        # Overwrite the payload with the correct fractional math before ingestion
        trade['price_deviation'] = unit_price / market_avg if market_avg > 0 else 1.0
        trade['weight_gap_score'] = weight_gap_raw / declared_weight if declared_weight > 0 else 0.0

    # 1. Save to Memgraph
    ingest_live_transaction(payload)
        
    # 2. Extract mathematically perfect tensors
    kyc_x, edge_index, swift_edge_attr, seq, trade_features = get_inference_tensors(tx_id, swift, trade)
    
    # 3. Forward Pass
    with torch.no_grad():
        logits = model(kyc_x, edge_index, swift_edge_attr, seq, trade_features)
        
    # 4. Classification
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0][predicted_class].item() * 100

    print(f"\n[INFERENCE REPORT] Transaction: {tx_id} | Amount: ${swift['amount']:,.2f}")
    if predicted_class == 2:
        print(f"🚨 VERDICT: [FRAUD] | Confidence: {confidence:.1f}%")
        print(f"   Action: Funds Frozen. Routing to compliance team.")
    elif predicted_class == 1:
        print(f"⚠️ VERDICT: [WATCHLIST] | Confidence: {confidence:.1f}%")
        print(f"   Action: Transaction held for manual audit.")
    else:
        print(f"✅ VERDICT: [CLEAN] | Confidence: {confidence:.1f}%")
        print(f"   Action: Cleared for settlement.")

# --- Start Listening ---
try:
    for message in consumer:
        process_live_transaction(message.value)
except KeyboardInterrupt:
    print("\n🛑 Shutting down Inference Engine.")
finally:
    consumer.close()
    db_driver.close()