import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATv2Conv
from neo4j import GraphDatabase
import argparse
from sklearn.metrics import f1_score
from collections import OrderedDict
from model import TBML_DetectionModel

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


# --- 3. MEMGRAPH DATA BRIDGE ---
def get_all_msg_ids(uri, user, password):
    """Automatically fetches all transaction IDs from this specific bank's Memgraph instance."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run("MATCH (t:Transaction) RETURN t.id AS msg_id LIMIT 100000" ) # Adjust limit as needed
        ids = [record["msg_id"] for record in result]
    driver.close()
    return ids

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
    def __init__(self, model, train_loader, test_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 20.0], device=self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        # Local Training Loop
        for epoch in range(3): # 3 local epochs per federated round
            for kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label in self.train_loader:
                # Squeeze the batch dimension for PyG compatibility
                kyc_x, edge_index = kyc_x[0], edge_index[0]
                swift_edge_attr, label = swift_edge_attr[0], label[0]
                trade_features = trade_features[0].unsqueeze(0)

                kyc_x = kyc_x.to(self.device)
                edge_index = edge_index.to(self.device)
                swift_edge_attr = swift_edge_attr.to(self.device)
                seq_data = seq_data.to(self.device)
                trade_features = trade_features.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(kyc_x, edge_index, swift_edge_attr, seq_data, trade_features)
                
                loss = self.criterion(logits, label.unsqueeze(0))
                loss.backward()
                self.optimizer.step()
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label in self.test_loader:
                kyc_x, edge_index = kyc_x[0], edge_index[0]
                swift_edge_attr, label = swift_edge_attr[0], label[0]
                trade_features = trade_features[0].unsqueeze(0)

                kyc_x = kyc_x.to(self.device)
                edge_index = edge_index.to(self.device)
                swift_edge_attr = swift_edge_attr.to(self.device)
                seq_data = seq_data.to(self.device)
                trade_features = trade_features.to(self.device)
                label = label.to(self.device)

                logits = self.model(kyc_x, edge_index, swift_edge_attr, seq_data, trade_features)
                loss = self.criterion(logits, label.unsqueeze(0))
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.append(label.cpu().item())

        accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        f1_m = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # STR Alert Check
        laundering_count = all_preds.count(2)
        if laundering_count > 0:
            print(f"\n🚨 STR ALERT [{args.bank.upper()}] - {laundering_count} Typologies detected in evaluation!")

        return float(total_loss / len(self.test_loader)), len(self.test_loader.dataset), {"accuracy": accuracy, "f1_macro": f1_m}

if __name__ == "__main__":
    print(f"🏦 Initializing {args.bank.upper()} on port {PORT_MAP[args.bank]}...")

    # 1. Dynamically pull all msg_ids from this bank's Memgraph container
    all_ids = get_all_msg_ids(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS)
    print(f"Loaded {len(all_ids)} transactions from Memgraph.")

    # 2. Split into Train (80%) and Test (20%)
    dataset = MemgraphTBMLDataset(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS, all_ids)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Note: batch_size=1 used here to safely stream individual subgraphs into PyG's GATv2Conv
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. Start Flower Client
    model = TBML_DetectionModel()
    client = BankClient(model, train_loader, test_loader)
    
    print(f"🚀 {args.bank.upper()} connecting to Central Server...")
    fl.client.start_numpy_client(server_address="127.0.0.1:8085", client=client)