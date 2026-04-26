"""
Real-Time AML Inference Pipeline v2.0
- Enhanced error handling & recovery
- Confidence scores & explainability
- Structured STR reports with audit trail
- Proper resource cleanup
"""

import json
import os
import uuid
import logging
import hashlib
import datetime
import torch
import numpy as np
from collections import OrderedDict
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'federated_learning')))
from model import TBML_DetectionModel

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(__file__)
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
KAFKA_TOPIC = "incoming_swift_messages"
MEMGRAPH_URI = "bolt://localhost:4000"
MEMGRAPH_USER = ""
MEMGRAPH_PASS = ""
CONFIDENCE_THRESHOLD = 0.65
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 16  # For future batch processing optimization

logger.info("🚀 Initializing Real-Time AML Inference Pipeline v2.0...")

# --- 2. LOAD THE GLOBAL FEDERATED MODEL ---
def load_model(model_path):
    """Load federated model weights with validation"""
    try:
        model = TBML_DetectionModel(kyc_dim=3, swift_dim=1, trade_dim=3)
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        logger.info(f"🧠 Loading Federated Global Weights from '{os.path.basename(model_path)}'...")
        params = np.load(model_path)
        
        # Validate parameter count
        model_params = list(model.state_dict().keys())
        loaded_params = list(params.keys())
        
        if len(model_params) != len(loaded_params):
            logger.warning(f"⚠️  Parameter mismatch: model has {len(model_params)}, "
                          f"loaded {len(loaded_params)}")
        
        # Reconstruct state dict with validation
        state_dict = OrderedDict()
        for key, array_val in zip(model_params, params.values()):
            if key not in model.state_dict():
                logger.warning(f"⚠️  Unknown parameter: {key}")
                continue
            state_dict[key] = torch.tensor(array_val)
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # Compute model hash for audit trail
        model_bytes = b''.join([arr.tobytes() for arr in params.values()])
        model_hash = hashlib.sha256(model_bytes).hexdigest()[:16]
        
        logger.info(f"✅ Model loaded successfully. Hash: {model_hash}")
        return model, model_hash
    
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        raise


# Load model
global_model_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'federated_learning', 'global_model.npz'))
model, model_hash = load_model(global_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f"Model running on: {device}")

# --- 3. MEMGRAPH DRIVER WITH RECONNECTION LOGIC ---
class MemgraphConnection:
    def __init__(self, uri, user, password, max_retries=MAX_RETRIES):
        self.uri = uri
        self.user = user
        self.password = password
        self.max_retries = max_retries
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection with retry logic"""
        for attempt in range(self.max_retries):
            try:
                auth = (self.user, self.password) if self.user else None
                self.driver = GraphDatabase.driver(self.uri, auth=auth)
                logger.info(f"✅ Connected to Memgraph at {self.uri}")
                return
            except (ServiceUnavailable, AuthError) as e:
                logger.warning(f"⚠️  Connection attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"❌ Failed to connect after {self.max_retries} attempts")
                    raise
    
    def query(self, cypher_string, **kwargs):
        """Execute query with error handling"""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_string, **kwargs)
                return result.single() if "RETURN" in cypher_string else result
        except ServiceUnavailable:
            logger.warning("⚠️  Database connection lost, attempting reconnect...")
            self._connect()
            return None
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return None
    
    def close(self):
        """Cleanup connection"""
        if self.driver:
            self.driver.close()
            logger.info("✅ Memgraph connection closed")


db_conn = MemgraphConnection(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS)


def fetch_graph_context(msg_id, max_retries=MAX_RETRIES):
    """
    Query Memgraph for sender, receiver, and trade document context.
    Returns None if transaction not found or on error.
    """
    query = """
    MATCH (sender:Account)-[:SENDS]->(t:Transaction {id: $msg_id})-[:TO]->(receiver:Account)
    OPTIONAL MATCH (t)-[:HAS_DOC]->(trade:Document)
    RETURN sender, receiver, t AS swift, trade
    """
    
    result = db_conn.query(query, msg_id=msg_id)
    
    if not result:
        logger.warning(f"⚠️  Transaction {msg_id} not found in graph database")
        return None
    
    return result


def compute_heuristic_scores(sender, receiver, trade):
    """
    Compute interpretable heuristic risk scores.
    Returns individual components for audit trail.
    """
    scores = {
        "w1_shell_dormancy": 0.0,
        "w2_tbml_indicators": 0.0,
        "w3_topology_risk": 0.0,
        "reasoning": []
    }
    
    # W1: Shell Corporation + Account Dormancy
    dorm_days = float(sender.get("dorm_days", 0))
    device_entropy = float(sender.get("device_entropy", 0))
    is_shell = float(sender.get("is_shell", 0))
    
    if is_shell > 0.5:
        scores["w1_shell_dormancy"] += 0.15
        scores["reasoning"].append("Sender flagged as high-risk shell entity")
    
    if dorm_days > 180 and device_entropy > 0.5:
        scores["w1_shell_dormancy"] += 0.25
        scores["reasoning"].append(
            f"High dormancy ({dorm_days} days) + device randomization: "
            f"Typical of layering/integration phase"
        )
    elif dorm_days > 90:
        scores["w1_shell_dormancy"] += 0.10
        scores["reasoning"].append(f"Elevated account dormancy: {dorm_days} days")
    
    # W2: Trade-Based ML Indicators
    if trade:
        price_dev = float(trade.get("price_deviation", 0))
        weight_gap = float(trade.get("weight_gap_score", 0))
        ais_status = trade.get("ais_status", "")
        
        # Over-invoicing detection
        if price_dev >= 3.0:
            scores["w2_tbml_indicators"] += 0.30
            scores["reasoning"].append(
                f"Extreme over-invoicing: {price_dev:.1f}x market value. "
                f"Typical value transfer mechanism."
            )
        elif price_dev > 1.5:
            scores["w2_tbml_indicators"] += 0.15
            scores["reasoning"].append(f"Suspicious pricing: {price_dev:.1f}x average")
        
        # Under-invoicing detection
        if price_dev < 0.3:
            scores["w2_tbml_indicators"] += 0.25
            scores["reasoning"].append(
                f"Extreme under-invoicing: {price_dev:.1f}x market. "
                f"Value outflow mechanism."
            )
        elif price_dev < 0.7:
            scores["w2_tbml_indicators"] += 0.10
            scores["reasoning"].append(f"Suspicious underpricing: {price_dev:.1f}x average")
        
        # Weight/quantity mismatches
        if weight_gap > 0.5:
            scores["w2_tbml_indicators"] += 0.10
            scores["reasoning"].append(
                f"Physical weight mismatch: {weight_gap:.1%} deviation. "
                f"Possible phantom shipment."
            )
        
        # Dark fleet activity
        if ais_status == "DARK":
            scores["w2_tbml_indicators"] += 0.15
            scores["reasoning"].append(
                "Vessel AIS transponder disabled during voyage: Dark Fleet pattern"
            )
    
    # W3: Network Topology Risk (Fan-out / Smurfing)
    try:
        sender_id = sender.get("id")
        result = db_conn.query(
            "MATCH (a:Account {id: $sid})-[:SENDS]->() RETURN count(*) as fan_out",
            sid=sender_id
        )
        fan_out = result["fan_out"] if result else 0
        
        if fan_out > 10:
            scores["w3_topology_risk"] = 0.35
            scores["reasoning"].append(
                f"Extreme fan-out pattern: {fan_out} outgoing transactions. "
                f"Structuring/Smurfing behavior."
            )
        elif fan_out > 5:
            scores["w3_topology_risk"] = 0.15
            scores["reasoning"].append(
                f"Elevated fan-out: {fan_out} transactions. "
                f"Possible distribution/structuring."
            )
    except Exception as e:
        logger.error(f"❌ Topology query failed: {e}")
    
    return scores


def write_str_to_db(msg_id, prediction_class, confidence, heuristic_scores, 
                    model_pred_probs, sender_id):
    """
    Write comprehensive STR Report to database with full audit trail.
    """
    str_id = f"STR-{uuid.uuid4().hex[:8].upper()}"
    
    # Map prediction to label
    class_labels = ["CLEAN", "SUSPICIOUS", "FRAUD"]
    pred_label = class_labels[prediction_class] if prediction_class < len(class_labels) else "UNKNOWN"
    
    # Combine heuristic reasoning
    primary_reason = heuristic_scores["reasoning"][0] if heuristic_scores["reasoning"] else "GNN pattern detected"
    
    # Write STR node
    query = """
    MATCH (t:Transaction {id: $msg_id})
    CREATE (str:STR_Report {
        id: $str_id,
        timestamp: timestamp(),
        classification: $classification,
        confidence: $confidence,
        model_version: $model_version,
        model_hash: $model_hash,
        heuristic_w1: $w1,
        heuristic_w2: $w2,
        heuristic_w3: $w3,
        heuristic_total: $total,
        decision_override: $override,
        primary_reason: $primary_reason,
        full_reasoning: $reasoning_list,
        model_probs_clean: $prob_clean,
        model_probs_suspicious: $prob_suspicious,
        model_probs_fraud: $prob_fraud
    })
    CREATE (str)-[:FLAGS]->(t)
    RETURN str.id AS str_id
    """
    
    try:
        heuristic_total = (heuristic_scores["w1_shell_dormancy"] + 
                          heuristic_scores["w2_tbml_indicators"] + 
                          heuristic_scores["w3_topology_risk"])
        
        result = db_conn.query(
            query,
            msg_id=msg_id,
            str_id=str_id,
            classification=pred_label,
            confidence=float(confidence),
            model_version="v2.0.0",
            model_hash=model_hash,
            w1=float(heuristic_scores["w1_shell_dormancy"]),
            w2=float(heuristic_scores["w2_tbml_indicators"]),
            w3=float(heuristic_scores["w3_topology_risk"]),
            total=float(heuristic_total),
            override=(heuristic_total >= 0.70),
            primary_reason=primary_reason,
            reasoning_list=heuristic_scores["reasoning"],
            prob_clean=float(model_pred_probs[0]) if len(model_pred_probs) > 0 else 0.0,
            prob_suspicious=float(model_pred_probs[1]) if len(model_pred_probs) > 1 else 0.0,
            prob_fraud=float(model_pred_probs[2]) if len(model_pred_probs) > 2 else 0.0
        )
        
        if result:
            logger.info(f"✅ STR Report created: {str_id}")
            return str_id
    except Exception as e:
        logger.error(f"❌ Failed to write STR report: {e}")
    
    return None


# --- 4. KAFKA STREAMING LOOP ---
def run_inference_pipeline():
    """Main inference loop"""
    logger.info(f"🎧 Connecting to Kafka broker: {KAFKA_BROKER}")
    logger.info(f"📋 Topic: {KAFKA_TOPIC}")
    
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            group_id='tbml_inference_v2'
        )
        logger.info("✅ Kafka consumer connected")
    except KafkaError as e:
        logger.error(f"❌ Kafka connection failed: {e}")
        raise

    logger.info("\n🟢 Pipeline Active. Waiting for real-time SWIFT transactions...\n")
    logger.info("-" * 80)

    processed = 0
    errors = 0

    try:
        for message in consumer:
            try:
                tx_data = message.value
                msg_id = tx_data.get("msg_id")
                
                logger.info(f"📥 Transaction: {msg_id} | Amount: ${tx_data.get('amount', 0)}")
                
                # 1. Fetch Graph Context
                context = fetch_graph_context(msg_id)
                if not context:
                    logger.warning(f"   ⚠️  Skipping {msg_id}: no context in database")
                    errors += 1
                    continue
                
                sender, receiver, swift, trade = (
                    context["sender"], context["receiver"], 
                    context["swift"], context["trade"]
                )

                # 2. Build Input Tensors
                kyc_x = torch.tensor([
                    [float(sender.get("is_shell", 0)), float(sender.get("dorm_days", 0)), 
                     float(sender.get("device_entropy", 0))],
                    [float(receiver.get("is_shell", 0)), float(receiver.get("dorm_days", 0)), 
                     float(receiver.get("device_entropy", 0))]
                ], dtype=torch.float).to(device)

                edge_index = torch.tensor([[0], [1]], dtype=torch.long).to(device)
                swift_edge_attr = torch.tensor(
                    [[float(swift.get("amount", tx_data.get("amount", 0)))]],
                    dtype=torch.float
                ).to(device)

                if trade:
                    ais_dark = 1.0 if trade.get("ais_status") == "DARK" else 0.0
                    trade_features = torch.tensor([
                        [float(trade.get("price_deviation", 0)),
                         float(trade.get("weight_gap_score", 0)),
                         ais_dark]
                    ], dtype=torch.float).to(device)
                else:
                    trade_features = torch.zeros((1, 3), dtype=torch.float).to(device)

                seq_data = swift_edge_attr.unsqueeze(0).to(device)  # [1, 1, 1]

                # 3. Model Forward Pass with Confidence
                with torch.no_grad():
                    logits = model(kyc_x, edge_index, swift_edge_attr, seq_data, trade_features)
                    probs = torch.softmax(logits, dim=1)
                    confidence = probs.max().item()
                    model_pred = probs.argmax(dim=1).item()
                    model_probs = probs.cpu().numpy()[0]

                # 4. Heuristic Scoring
                heuristic_scores = compute_heuristic_scores(sender, receiver, trade)
                heuristic_total = (heuristic_scores["w1_shell_dormancy"] + 
                                 heuristic_scores["w2_tbml_indicators"] + 
                                 heuristic_scores["w3_topology_risk"])

                # 5. Hybrid Decision Logic
                final_prediction = model_pred
                if model_pred == 0 and heuristic_total >= 0.70:
                    final_prediction = 1
                    logger.info(f"   ⚠️ Override: GNN said CLEAN but heuristic score ({heuristic_total:.2f}) forced SUSPICIOUS")
                
                pred_class_names = ["CLEAN", "SUSPICIOUS", "FRAUD"]
                pred_label = pred_class_names[final_prediction]
                
                logger.info(f"   📊 GNN: {pred_class_names[model_pred]} ({confidence:.1%}) | "
                          f"Heuristic: {heuristic_total:.2f} → Final: {pred_label}")

                # 6. Action / Output
                if final_prediction > 0:
                    str_id = write_str_to_db(
                        msg_id, final_prediction, confidence, heuristic_scores,
                        model_probs, sender.get("id")
                    )
                    
                    if final_prediction == 2:
                        logger.warning(f"   🚨 FRAUD DETECTED (STR: {str_id})")
                    else:
                        logger.warning(f"   🟡 SUSPICIOUS (STR: {str_id})")
                    
                    logger.info("   📋 Risk Factors:")
                    for reason in heuristic_scores["reasoning"]:
                        logger.info(f"      • {reason}")
                else:
                    logger.info("   🟢 CLEAR")

                processed += 1
                logger.info("-" * 80)

            except Exception as e:
                logger.error(f"❌ Error processing transaction: {e}", exc_info=True)
                errors += 1
                continue

    finally:
        consumer.close()
        db_conn.close()
        logger.info(f"\n✅ Pipeline shutdown. Processed: {processed}, Errors: {errors}")


if __name__ == "__main__":
    try:
        run_inference_pipeline()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Pipeline stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal pipeline error: {e}", exc_info=True)
        raise
