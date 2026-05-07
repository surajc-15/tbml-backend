import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATv2Conv
from neo4j import GraphDatabase
import argparse
import time
from collections import OrderedDict
import sys
import random # --- ADDED FOR DATA SHUFFLING ---
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import TBML_DetectionModel
from utilities.logger import log_event, setup_logger

# --- IMPORTS FOR GRAPHING & METRICS ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

# --- 1. CONFIG & DOCKER PORT MAPPING ---
parser = argparse.ArgumentParser()
parser.add_argument("--bank", type=str, required=True, choices=['banka', 'bankb', 'bankc'])
args = parser.parse_args()

# Map the bank argument to your specific Docker ports
PORT_MAP = {
    "banka": 4000,
    "bankb": 4001,
    "bankc": 4002
}
MEMGRAPH_URI = f"bolt://localhost:{PORT_MAP[args.bank]}"
# Replace with actual auth if you set it in Docker, otherwise leave empty
MEMGRAPH_USER = ""
MEMGRAPH_PASS = ""
SAMPLE_SIZE = 800000
BATCH_SIZE = 16
LOCAL_EPOCHS = 3
logger = setup_logger("tbml.client", use_color=False)


# --- 2. GRAPH GENERATOR HELPER FUNCTION ---
def generate_presentation_metrics(y_true, y_pred, y_probs, bank_name):
    classes = [0, 1, 2]
    class_names = ['Clean (0)', 'Watchlist (1)', 'Fraud (2)']
    
    print(f"\n--- {bank_name.upper()} FINAL EVALUATION METRICS ---")
    print(classification_report(y_true, y_pred, labels=classes, target_names=class_names, zero_division=0))
    # 1. Generate and Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{bank_name.upper()} - TBML Confusion Matrix')
    plt.ylabel('Actual Truth')
    plt.xlabel('AI Prediction')
    plt.tight_layout()
    plt.savefig(f'{bank_name}_confusion_matrix.png', dpi=300)
    plt.close()

    # 2. Generate and Save ROC Curve (Focusing on the Fraud Class)
    y_true_bin = label_binarize(y_true, classes=classes)
    fraud_class_idx = 2
    
    if np.sum(y_true_bin[:, fraud_class_idx]) > 0:
        fpr, tpr, _ = roc_curve(y_true_bin[:, fraud_class_idx], y_probs[:, fraud_class_idx])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Fraud ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{bank_name.upper()} - Receiver Operating Characteristic (Fraud)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'{bank_name}_roc_curve.png', dpi=300)
        plt.close()

    # 3. Generate and Save Precision, Recall, and F1 Score Bar Chart
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    rects2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')

    ax.set_ylabel('Score (0.0 to 1.0)')
    ax.set_title(f'{bank_name.upper()} - Model Performance by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim([0.0, 1.1]) 
    ax.legend(loc='upper right')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(f'{bank_name}_performance_metrics.png', dpi=300)
    plt.close()


# --- 3. MEMGRAPH DATA BRIDGE ---
def get_all_msg_ids(uri, user, password, sample_size=150000):
    """Fetches all IDs and safely shuffles them in Python to ensure Fraud representation."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # 1. Pull ALL transaction IDs (pulling just the string IDs is very fast and low-memory)
        log_event(logger, headline="Fetching transaction IDs", status="INFO", message="Querying Memgraph for transaction IDs")
        result = session.run("MATCH (t:Transaction) RETURN t.id AS msg_id")
        ids = [record["msg_id"] for record in result]
    driver.close()

    # 2. Randomly shuffle the deck so Fraud is distributed evenly
    log_event(logger, headline="Shuffling transaction IDs", status="INFO", message="Randomizing sampled IDs before split")
    random.shuffle(ids)

    # 3. Take a safe slice (150k) to train on
    return ids[:sample_size]

class MemgraphTBMLDataset(Dataset):
    def __init__(self, uri, user, password, transaction_ids):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.transaction_ids = transaction_ids

    def __len__(self):
        return len(self.transaction_ids)

    def __getitem__(self, idx):
        target_msg_id = self.transaction_ids[idx]
        
        query = """
        MATCH (sender:Account)-[:SENDS]->(t:Transaction {id: $target_msg_id})-[:TO]->(receiver:Account)
        OPTIONAL MATCH (t)-[:HAS_DOC]->(trade:Document)
        RETURN sender, receiver, t AS swift, trade
        """
        
        with self.driver.session() as session:
            result = session.run(query, target_msg_id=target_msg_id).single()

        sender, receiver, swift, trade = result["sender"], result["receiver"], result["swift"], result["trade"]

        kyc_x = torch.tensor([
            [float(sender.get("is_shell", 0)), float(sender.get("dorm_days", 0)), float(sender.get("device_entropy", 0))],
            [float(receiver.get("is_shell", 0)), float(receiver.get("dorm_days", 0)), float(receiver.get("device_entropy", 0))]
        ], dtype=torch.float)

        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        swift_edge_attr = torch.tensor([[float(swift.get("amount", 0))]], dtype=torch.float)

        if trade:
            ais_dark = 1.0 if trade.get("ais_status") == "DARK" else 0.0
            trade_features = torch.tensor([
                float(trade.get("price_deviation", 0)),
                float(trade.get("weight_gap_score", 0)),
                ais_dark
            ], dtype=torch.float)
        else:
            trade_features = torch.zeros(3, dtype=torch.float)

        seq_data = swift_edge_attr.unsqueeze(0) 
        label = torch.tensor(int(swift.get("label", 0)), dtype=torch.long)

        return kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label

# --- 4. FLOWER FEDERATED CLIENT ---
class BankClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.device = device
        # Move model to the target device first
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Ensure loss weights live on the same device
        weights = torch.tensor([1.0, 10.0, 20.0], device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        # Create optimizer after model is on the device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
        # Ensure tensors are moved to cpu when loading into state_dict if needed by PyTorch
        # but keep dtype/device consistent by loading directly (model is already on device)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        train_samples = len(self.train_loader.dataset)
        total_batches = len(self.train_loader)
        log_event(logger, headline="Training started", status="INFO", message=f"Starting {LOCAL_EPOCHS} epochs on {train_samples} samples with batch size {BATCH_SIZE}")
        
        # Local Training Loop
        overall_start = time.perf_counter()
        for epoch in range(LOCAL_EPOCHS):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            epoch_total = 0
            for batch_idx, (kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label) in enumerate(self.train_loader, start=1):
                # Squeeze the batch dimension for PyG compatibility and move to device
                kyc_x, edge_index = kyc_x[0].to(self.device), edge_index[0].to(self.device)
                swift_edge_attr, label = swift_edge_attr[0].to(self.device), label[0].to(self.device)
                seq_data = seq_data[0].to(self.device)
                trade_features = trade_features[0].unsqueeze(0).to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(kyc_x, edge_index, swift_edge_attr, seq_data, trade_features)

                loss = self.criterion(logits, label.unsqueeze(0))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_total += 1

                if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(total_batches // 10, 1) == 0:
                    epoch_pct = (batch_idx / max(total_batches, 1)) * 100.0
                    sample_pct = (epoch_total / max(train_samples, 1)) * 100.0
                    log_event(
                        logger,
                        headline=f"Epoch {epoch + 1} progress",
                        status="INFO",
                        message=f"Epoch {epoch + 1}/{LOCAL_EPOCHS}: {batch_idx}/{total_batches} batches complete ({epoch_pct:.1f}%), {epoch_total}/{train_samples} samples seen ({sample_pct:.1f}%), last loss {loss.item():.6f}",
                    )

            log_event(
                logger,
                headline=f"Epoch {epoch + 1} finished",
                status="SUCCESS",
                message=f"Epoch {epoch + 1}/{LOCAL_EPOCHS} done. Average loss {epoch_loss / max(epoch_total, 1):.6f}. Time this epoch {time.perf_counter() - epoch_start:.1f}s. Total training time {time.perf_counter() - overall_start:.1f}s.",
            )
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        log_event(logger, headline="Evaluation started", status="INFO", message=f"Evaluating on {len(self.test_loader.dataset)} samples with batch size {BATCH_SIZE}")
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = [] # ADDED to capture raw probabilities for ROC curve

        with torch.no_grad():
            for kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label in self.test_loader:
                kyc_x, edge_index = kyc_x[0].to(self.device), edge_index[0].to(self.device)
                swift_edge_attr, label = swift_edge_attr[0].to(self.device), label[0].to(self.device)
                seq_data = seq_data[0].to(self.device)
                trade_features = trade_features[0].unsqueeze(0).to(self.device)

                logits = self.model(kyc_x, edge_index, swift_edge_attr, seq_data, trade_features)
                loss = self.criterion(logits, label.unsqueeze(0))

                total_loss += loss.item()

                # Apply Softmax to get probabilities (0.0 to 1.0)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.append(label.cpu().item())

        # Extract metrics and trigger the graph generation
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)
        generate_presentation_metrics(y_true, y_pred, y_probs, args.bank)

        accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        f1_m = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # STR Alert Check
        laundering_count = all_preds.count(2)
        if laundering_count > 0:
            log_event(
                logger,
                headline="STR alert",
                status="WARNING",
                message=f"Detected {laundering_count} suspicious typologies during evaluation.",
            )

        return float(total_loss / len(self.test_loader)), len(self.test_loader.dataset), {"accuracy": accuracy, "f1_macro": f1_m}

if __name__ == "__main__":
    log_event(logger, headline="Client initialization", status="INFO", message=f"Starting client for {args.bank.upper()} on port {PORT_MAP[args.bank]}")

    # 1. Dynamically pull and shuffle msg_ids
    all_ids = get_all_msg_ids(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS, sample_size=SAMPLE_SIZE)
    log_event(logger, headline="Data ready", status="SUCCESS", message=f"Loaded and shuffled {len(all_ids)} transaction IDs from Memgraph (sample size {SAMPLE_SIZE})")

    # 2. Split into Train (80%) and Test (20%)
    dataset = MemgraphTBMLDataset(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS, all_ids)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    log_event(logger, headline="Configuration", status="INFO", message=f"Train split: {len(train_dataset)} samples, test split: {len(test_dataset)} samples, batch size {BATCH_SIZE}, epochs {LOCAL_EPOCHS}")

    # 3. Device setup and start Flower Client
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_event(logger, headline="Device selected", status="INFO", message=f"Using {device} for training")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = TBML_DetectionModel()
    client = BankClient(model, train_loader, test_loader, device)
    
    log_event(logger, headline="Connecting", status="INFO", message="Connecting to Flower server at 127.0.0.1:8085")
    fl.client.start_numpy_client(server_address="127.0.0.1:8085", client=client)