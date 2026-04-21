import json
import os
import uuid
import datetime
import torch
import numpy as np
from collections import OrderedDict
from neo4j import GraphDatabase
from kafka import KafkaConsumer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'federated_learning')))
from model import TBML_DetectionModel

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(__file__)
# docker-compose maps Kafka to localhost:19092; override if you run Kafka elsewhere.
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
KAFKA_TOPIC = "incoming_swift_messages"
MEMGRAPH_URI = "bolt://localhost:4000"  # We'll use Bank A's Memgraph for context
MEMGRAPH_USER = ""
MEMGRAPH_PASS = ""

print("🚀 Initializing Real-Time AML Inference Pipeline...")

# --- 2. LOAD THE GLOBAL FEDERATED MODEL ---
# Initialize the blank architecture
model = TBML_DetectionModel(kyc_dim=3, swift_dim=1, trade_dim=3)

# Load the weights from the .npz file
print("🧠 Loading Federated Global Weights from 'global_model.npz'...")
params = np.load(os.path.abspath(os.path.join(BASE_DIR, '..', 'federated_learning', 'global_model.npz')))

# Reconstruct the PyTorch state dictionary from the NumPy arrays
state_dict = OrderedDict()
for key, array_val in zip(model.state_dict().keys(), params.values()):
    state_dict[key] = torch.tensor(array_val)

model.load_state_dict(state_dict, strict=True)
model.eval()  # Set model to evaluation mode
print("✅ Brain loaded successfully.")

# --- 3. MEMGRAPH DRIVER ---
driver = GraphDatabase.driver(MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASS))

def fetch_graph_context(msg_id):
    """Queries Memgraph for the sender, receiver, and trade document context."""
    query = """
    MATCH (sender:Account)-[:SENDS]->(t:Transaction {id: $msg_id})-[:TO]->(receiver:Account)
    OPTIONAL MATCH (t)-[:HAS_DOC]->(trade:Document)
    RETURN sender, receiver, t AS swift, trade
    """
    with driver.session() as session:
        result = session.run(query, msg_id=msg_id).single()
    return result

def generate_reasons(sender, receiver, trade):
    reasons = []
    if float(sender.get("is_shell", 0)) == 1.0:
        reasons.append("Sender flagged as high-risk shell corporation.")
    if float(sender.get("dorm_days", 0)) > 30:
        reasons.append(f"Sender account exhibited unusual dormancy ({sender.get('dorm_days')} days).")
        
    if trade:
        price_dev = float(trade.get("price_deviation", 0))
        if price_dev > 1.5:
            reasons.append(f"Trade Overpricing Detected: Goods are {price_dev:.1f}x the market average.")
        elif price_dev < 0.5:
            reasons.append(f"Trade Underpricing Detected: Goods are artificially discounted to funnel value.")
            
        if float(trade.get("weight_gap_score", 0)) > 0.5:
            reasons.append("Discrepancy between declared weight and actual AIS physical displacement weight.")
            
        if trade.get("ais_status") == "DARK":
            reasons.append("Vessel tracker (AIS) was turned off during voyage (Dark Fleet activity).")
            
    return reasons if reasons else ["Suspicious network topology detected by GNN."]

def write_str_to_db(msg_id, prediction_class, reasons):
    str_id = f"STR-{uuid.uuid4().hex[:8].upper()}"
    query = """
    MATCH (t:Transaction {id: $msg_id})
    CREATE (str:STR_Report {
        id: $str_id,
        classification_level: $pred_class,
        reasons: $reasons_list,
        timestamp: timestamp()
    })
    CREATE (str)-[:FLAGS]->(t)
    RETURN str.id AS str_id
    """
    with driver.session() as session:
        result = session.run(query, 
                             msg_id=msg_id, 
                             str_id=str_id,
                             pred_class=prediction_class, 
                             reasons_list=reasons).single()
        if result:
            return result["str_id"]
    return None

# --- 4. KAFKA STREAMING LOOP ---
print(f"🎧 Subscribing to Kafka topic: '{KAFKA_TOPIC}'...")
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest'
)

print("\n🟢 Pipeline Active. Waiting for real-time SWIFT transactions...\n")
print("-" * 60)

for message in consumer:
    tx_data = message.value
    msg_id = tx_data.get("msg_id")
    
    print(f"📥 New Transaction Received: {msg_id} | Amount: ${tx_data.get('amount', 0)}")
    
    # 1. Fetch Graph Context from Memgraph
    context = fetch_graph_context(msg_id)
    if not context:
        print(f"   ⚠️ WARNING: No graph context found in Memgraph for {msg_id}. Skipping.")
        print("-" * 60)
        continue
        
    sender, receiver, swift, trade = context["sender"], context["receiver"], context["swift"], context["trade"]

    # 2. Convert to PyTorch Tensors
    kyc_x = torch.tensor([
        [float(sender.get("is_shell", 0)), float(sender.get("dorm_days", 0)), float(sender.get("device_entropy", 0))],
        [float(receiver.get("is_shell", 0)), float(receiver.get("dorm_days", 0)), float(receiver.get("device_entropy", 0))]
    ], dtype=torch.float)

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    swift_edge_attr = torch.tensor([[float(swift.get("amount", tx_data.get("amount", 0)))]], dtype=torch.float)

    if trade:
        ais_dark = 1.0 if trade.get("ais_status") == "DARK" else 0.0
        trade_features = torch.tensor([[  # Wrapped in extra bracket for batch dim
            float(trade.get("price_deviation", 0)),
            float(trade.get("weight_gap_score", 0)),
            ais_dark
        ]], dtype=torch.float)
    else:
        trade_features = torch.zeros((1, 3), dtype=torch.float)

    # Sequence data needs shape [batch, seq_len, features] -> [1, 1, 1]
    seq_data = swift_edge_attr.unsqueeze(0)

    # 3. Model Forward Pass & Hybrid Logic
    with torch.no_grad():
        logits = model(kyc_x, edge_index, swift_edge_attr, seq_data, trade_features)
        print("values recieved are ", logits, "and the argumenrts after converted are \n", kyc_x, "\n", edge_index, "\n", swift_edge_attr, "\n", seq_data, "\n", trade_features)
        model_pred = torch.argmax(logits, dim=1).item()

    # Calculate heuristic score for explanations and override logic
    w1 = 0.0
    if float(sender.get("dorm_days", 0)) > 180 and float(sender.get("device_entropy", 0)) > 0.5:
        w1 = 0.25
    elif float(sender.get("dorm_days", 0)) > 90:
        w1 = 0.10
        
    w2 = 0.0
    if trade:
        dev = float(trade.get("price_deviation", 0))
        wg = float(trade.get("weight_gap_score", 0))
        if dev >= 3.0: w2 += 0.20
        elif dev > 1.5: w2 += 0.10
        if wg >= 0.15: w2 += 0.10
        if trade.get("ais_status") == "DARK": w2 += 0.10
        
    w3 = 0.0
    with driver.session() as tp_session:
        # Proxy for Topology Fan-out/Cycle frequency
        top_res = tp_session.run("MATCH (a:Account {id: $sid})-[:SENDS]->(t) RETURN count(t) as c", sid=sender.get("id")).single()
        if top_res and top_res["c"] > 5:
            w3 = 0.35
        elif top_res and top_res["c"] > 2:
            w3 = 0.15

    score = w1 + w2 + w3
    
    # Hybrid Prediction Logic
    if model_pred == 0 and score >= 0.7:
        prediction = 1
        print(f"   📊 AI Prediction: CLEAN (0), but Heuristic Score ({score:.2f}) overrode to SUSPICIOUS (1).")
    else:
        prediction = model_pred
        class_str = ["CLEAN (0)", "SUSPICIOUS (1)", "FRAUD (2)"][prediction]
        print(f"   📊 AI Prediction: {class_str} | Heuristic Score: {score:.2f}")

    # 4. Action / Output
    if prediction > 0:
        reasons = generate_reasons(sender, receiver, trade)
        str_id = write_str_to_db(msg_id, prediction, reasons)
        
        if prediction == 2:
            print(f"   🚨 [STR ALERT] - High Prob TBML (STR ID: {str_id})")
        elif prediction == 1:
            print(f"   🟡 [REVIEW] - Suspicious Activity (STR ID: {str_id})")
            
        print("   🔍 REASONS LOGGED IN GRAPH:")
        for idx, r in enumerate(reasons):
            print(f"      {idx+1}. {r}")
    else:
        print("   🟢 [CLEAN] - Transaction cleared.")
        
    print("-" * 60)