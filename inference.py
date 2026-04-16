import json
import os
import torch
import numpy as np
from collections import OrderedDict
from neo4j import GraphDatabase
from kafka import KafkaConsumer
from model import TBML_DetectionModel

# --- 1. CONFIGURATION ---
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
params = np.load("global_model.npz")

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

# --- 4. KAFKA STREAMING LOOP ---
print(f"🎧 Subscribing to Kafka topic: '{KAFKA_TOPIC}'...")
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest'  # Only read new messages
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

    # 3. Model Inference (No gradients needed)
    with torch.no_grad():
        logits = model(kyc_x, edge_index, swift_edge_attr, seq_data, trade_features)
        prediction = logits.argmax(dim=1).item()

    # 4. Action / Output
    if prediction == 2:
        print("   🚨 [STR ALERT] - High Probability of Trade-Based Money Laundering Detected!")
    elif prediction == 1:
        print("   🟡 [REVIEW] - Suspicious patterns found. Flagged for manual review.")
    else:
        print("   🟢 [CLEAN] - Transaction cleared.")
        
    print("-" * 60)