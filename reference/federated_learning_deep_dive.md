# Federated Learning Deep Dive

## Purpose of This Document
This document explains the federated learning part of the TBML project in detail: the client, the server, how the global model is generated, where parameters come from, what each important value means, and how the pieces interact step by step.

This is based on the current code in:
- [federated_learning/client.py](../federated_learning/client.py)
- [federated_learning/server.py](../federated_learning/server.py)
- [federated_learning/model.py](../federated_learning/model.py)
- [federated_learning/tbml/models/detection.py](../federated_learning/tbml/models/detection.py)

---

## 1. What Federated Learning Means Here

Federated learning means each bank trains the same model locally on its own private graph data, and only the model parameters are shared with the central server.

### In this project
- Bank A runs one local client process.
- Bank B runs one local client process.
- Bank C runs one local client process.
- The central server receives model parameters from all three clients.
- The server averages the parameters using FedAvg.
- The updated global parameters are sent back to the clients for the next round.

### What is not shared
- Raw KYC CSV rows.
- Raw transaction graph context.
- Trade document records.
- Graph database contents.

### What is shared
- Neural network weights.
- Biases.
- Optimizer-independent model parameters.
- Evaluation metrics if returned by clients.

This is the privacy-preserving part of the system.

---

## 2. High-Level Flow

```text
Bank graph data -> local Memgraph -> client dataset -> local model training
                        |
                        v
                 local model weights
                        |
                        v
                   Flower server
                        |
                        v
                 federated averaging
                        |
                        v
                 global model weights
                        |
                        v
            saved as global_model.npz
                        |
                        v
         loaded by inference pipeline later
```

---

## 3. Current Code Architecture

### Client file
[client.py](../federated_learning/client.py) does 4 major jobs:
1. Connects to the bank-specific graph database.
2. Builds a dataset of transaction-centered samples.
3. Trains the TBML model locally.
4. Sends weights and metrics back to the server.

### Server file
[server.py](../federated_learning/server.py) does 3 major jobs:
1. Creates the initial model.
2. Starts a Flower FedAvg server.
3. Saves aggregated weights as `global_model.npz` at round 10.

### Model file
[model.py](../federated_learning/model.py) is just a wrapper import.
It re-exports `TBML_DetectionModel` from the package:
- [federated_learning/tbml/models/detection.py](../federated_learning/tbml/models/detection.py)

---

## 4. What the Client Actually Does

The client is the local bank training process.

### 4.1 Imports
```python
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
```

#### Meaning
- `flwr`: federated learning framework.
- `torch`: deep learning tensor library.
- `Dataset`, `DataLoader`, `random_split`: local data loading.
- `GraphDatabase`: connects to Memgraph/Neo4j-like Bolt database.
- `OrderedDict`: reconstructs model state from arrays.
- `f1_score`: evaluation metric.
- `TBML_DetectionModel`: the actual classifier architecture.

#### Impact
These imports define the whole local training pipeline.

---

### 4.2 Command-Line Argument Parsing
```python
parser = argparse.ArgumentParser()
parser.add_argument("--bank", type=str, required=True, choices=['banka', 'bankb', 'bankc'])
args = parser.parse_args()
```

#### What it does
The client requires you to specify which bank it represents.

#### Parameter details
- `--bank`
  - Type: string
  - Allowed values: `banka`, `bankb`, `bankc`
  - Required: yes

#### Why it matters
This selects the bank-specific database port and graph silo.

#### Impact
Each client sees only one bank's private graph data.

---

### 4.3 Bank-to-Port Mapping
```python
PORT_MAP = {
    "banka": 4000,
    "bankb": 4001,
    "bankc": 4002
}
MEMGRAPH_URI = f"bolt://localhost:{PORT_MAP[args.bank]}"
```

#### Meaning
Each bank uses a different local Memgraph port.

#### Parameter details
- `4000`, `4001`, `4002`
  - These are Bolt ports.
  - One port per bank.

#### Impact
- Bank A client connects to `bolt://localhost:4000`
- Bank B client connects to `bolt://localhost:4001`
- Bank C client connects to `bolt://localhost:4002`

This keeps the silos separated.

---

### 4.4 Graph Database Authentication Fields
```python
MEMGRAPH_USER = ""
MEMGRAPH_PASS = ""
```

#### Meaning
The code currently assumes no auth.

#### Range / values
- Empty strings mean anonymous/local access.
- In a real deployment, these should be replaced with secure credentials.

#### Impact
- Simple local development.
- Not ideal for production security.

---

## 5. Dataset Construction on the Client

### 5.1 get_all_msg_ids
```python
def get_all_msg_ids(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run("MATCH (t:Transaction) RETURN t.id AS msg_id LIMIT 100000")
        ids = [record["msg_id"] for record in result]
    driver.close()
    return ids
```

#### What it does
This function fetches all transaction IDs from the bank's graph database.

#### Inputs
- `uri`: Bolt URI of the bank database.
- `user`, `password`: auth credentials.

#### Query explanation
```cypher
MATCH (t:Transaction) RETURN t.id AS msg_id LIMIT 100000
```
- Finds all `Transaction` nodes.
- Returns only their `id` values.
- Stops at 100,000 rows.

#### Why `LIMIT 100000`
This prevents accidental huge loads.

#### Impact
- Sets the maximum local training set size.
- Controls memory and runtime.

#### Data source
The IDs come directly from the bank's graph, not from CSV files in the client.

---

### 5.2 MemgraphTBMLDataset
This class converts each transaction into one training sample.

```python
class MemgraphTBMLDataset(Dataset):
    def __init__(self, uri, user, password, transaction_ids):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.transaction_ids = transaction_ids
```

#### What it stores
- `self.driver`: database connection.
- `self.transaction_ids`: the list of transaction IDs to index.

#### Impact
This lets the dataset lazily fetch one graph sample at a time.

---

### 5.3 __len__
```python
def __len__(self):
    return len(self.transaction_ids)
```

#### Meaning
Dataset length equals the number of transaction IDs.

#### Impact
This controls how many samples the DataLoader can iterate over.

---

### 5.4 __getitem__
This is the most important function in the client dataset.

```python
def __getitem__(self, idx):
    target_msg_id = self.transaction_ids[idx]
```

#### What it does
Chooses one transaction ID as the current sample.

#### Input
- `idx`: integer position in the list.

#### Output
- A single sample representing one transaction and its local graph context.

---

#### Graph query used
```cypher
MATCH (sender:Account)-[:SENDS]->(t:Transaction {id: $target_msg_id})-[:TO]->(receiver:Account)
OPTIONAL MATCH (t)-[:HAS_DOC]->(trade:Document)
RETURN sender, receiver, t AS swift, trade
```

#### What this query returns
- `sender`: sender account node.
- `receiver`: receiver account node.
- `swift`: the transaction node.
- `trade`: linked trade document, if present.

#### Why it matters
This is the actual training evidence for the model.

#### Data source origin
- `sender` and `receiver` come from `Account` nodes.
- `swift` comes from `Transaction` nodes.
- `trade` comes from `Document` nodes.

---

### 5.5 KYC feature tensor
```python
kyc_x = torch.tensor([
    [float(sender.get("is_shell", 0)), float(sender.get("dorm_days", 0)), float(sender.get("device_entropy", 0))],
    [float(receiver.get("is_shell", 0)), float(receiver.get("dorm_days", 0)), float(receiver.get("device_entropy", 0))]
], dtype=torch.float)
```

#### Shape
- Shape: `[2, 3]`
- 2 nodes: sender and receiver
- 3 features each

#### Features
1. `is_shell`
   - Usually 0 or 1
   - Indicates shell-company risk
2. `dorm_days`
   - Number of dormant days
   - Usually non-negative integer/float
3. `device_entropy`
   - Measures device behavior variability
   - Usually 0 to 1 in your synthetic data

#### Impact
This is the graph node feature input to the GNN.

---

### 5.6 Edge index
```python
edge_index = torch.tensor([[0], [1]], dtype=torch.long)
```

#### Shape
- `[2, 1]`

#### Meaning
This means node 0 points to node 1, i.e. sender -> receiver.

#### Why it matters
The GNN uses edge structure to learn relational behavior.

#### Impact
Without edge information, the model would not learn transaction topology.

---

### 5.7 SWIFT edge attribute
```python
swift_edge_attr = torch.tensor([[float(swift.get("amount", 0))]], dtype=torch.float)
```

#### Shape
- `[1, 1]`

#### Feature
- `amount`

#### Range
- Depends on transaction size.
- In practice, could be small or huge.

#### Impact
This becomes the edge feature used by GATv2Conv and the sequence branch.

---

### 5.8 Trade document features
```python
if trade:
    ais_dark = 1.0 if trade.get("ais_status") == "DARK" else 0.0
    trade_features = torch.tensor([
        float(trade.get("price_deviation", 0)),
        float(trade.get("weight_gap_score", 0)),
        ais_dark
    ], dtype=torch.float)
else:
    trade_features = torch.zeros(3, dtype=torch.float)
```

#### Shape
- `[3]`

#### Features
1. `price_deviation`
   - Ratio of unit price to market average
   - Values > 1 suggest overpricing
   - Values < 1 suggest underpricing
2. `weight_gap_score`
   - Amount of mismatch between declared and actual weight
   - Higher values indicate suspicious shipment mismatch
3. `ais_dark`
   - 1 if AIS is `DARK`
   - 0 otherwise

#### If no trade document exists
- The vector becomes all zeros.

#### Impact
This branch captures TBML document anomalies.

---

### 5.9 Sequence data
```python
seq_data = swift_edge_attr.unsqueeze(0)
```

#### Shape
- From `[1, 1]` to `[1, 1, 1]`

#### Meaning
A fake one-step sequence is created around amount information.

#### Why it exists
The model contains an LSTM branch and expects sequence-shaped input.

#### Impact
It adds a temporal component, even if the current sequence is very simple.

---

### 5.10 Label
```python
label = torch.tensor(int(swift.get("label", 0)), dtype=torch.long)
```

#### Meaning
The target class for this sample.

#### Range
Usually:
- `0` = CLEAN
- `1` = SUSPICIOUS
- `2` = FRAUD

#### Default behavior
If no label exists, it defaults to `0`.

#### Impact
This is very important: if labels are missing, the model may learn a strong CLEAN bias.

---

### 5.11 Dataset return value
```python
return kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label
```

#### The sample contains
- graph node features
- graph connectivity
- edge amount
- sequence input
- trade features
- supervision label

This is the full local training example used by the client.

---

## 6. Local Training on the Client

### 6.1 BankClient class
```python
class BankClient(fl.client.NumPyClient):
```

#### Meaning
This is Flower’s NumPy client interface.

#### Why NumPyClient
It allows Flower to pass model parameters as NumPy arrays across the network.

#### Impact
This is what makes federated parameter exchange simple.

---

### 6.2 Constructor
```python
def __init__(self, model, train_loader, test_loader):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.5, 2.0]))
    self.optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### Parameters explained

##### `model`
- Instance of `TBML_DetectionModel`.
- Contains GAT + LSTM + trade MLP + classifier.

##### `train_loader`
- DataLoader for local training data.

##### `test_loader`
- DataLoader for local evaluation data.

##### `criterion`
- Cross-entropy loss with class weights.

##### `optimizer`
- Adam optimizer.
- Learning rate = `0.001`.

#### Class weights `[0.1, 1.5, 2.0]`
- Clean class gets low weight.
- Suspicious gets medium weight.
- Fraud gets highest weight.

#### Impact
This tells the model to care more about fraud mistakes than clean predictions.

#### Risk
If these weights do not match your actual data imbalance, the model may overfit or underfit one class.

---

### 6.3 get_parameters
```python
def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
```

#### What it does
Converts all PyTorch tensors in the model state dictionary into NumPy arrays.

#### Source of parameters
Directly from `self.model.state_dict()`.

#### What this includes
All trainable tensors, such as:
- GAT weights and biases
- LSTM weights and biases
- Trade MLP weights and biases
- Classifier weights and biases

#### Impact
These are the exact tensors sent to the Flower server.

---

### 6.4 set_parameters
```python
def set_parameters(self, parameters):
    params_dict = zip(self.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    self.model.load_state_dict(state_dict, strict=True)
```

#### What it does
Loads server-provided parameters back into the model.

#### How matching works
- `model.state_dict().keys()` gives parameter names in order.
- `parameters` is the list returned by Flower.
- `zip(...)` matches names with arrays.

#### Why `OrderedDict`
The order must remain consistent so the right weights go into the right layers.

#### Impact
If order mismatches, the model weights will be corrupted.

---

### 6.5 fit
```python
def fit(self, parameters, config):
```

#### Purpose
Perform local training for one federated round.

---

#### Step 1: receive global parameters
```python
self.set_parameters(parameters)
```
The client starts from the latest global model sent by the server.

---

#### Step 2: set training mode
```python
self.model.train()
```
Enables training behavior such as dropout.

---

#### Step 3: local epochs
```python
for epoch in range(3):
```

##### Meaning
The client trains for 3 local epochs per federated round.

##### Range
- Current value: `3`
- Allowed range: any positive integer

##### Impact
- Higher values: more local adaptation, slower communication rounds.
- Lower values: less local learning, faster rounds.

##### Why 3 is used
It is a compromise between local learning and federated synchronization.

---

#### Step 4: batch loop
```python
for kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label in self.train_loader:
```

##### Current batch size
From the DataLoader later in the code:
- `batch_size=1`

##### Why batch size 1
Graph samples are small and simpler to handle one at a time.

##### Impact
- Easy shape handling.
- Slower than larger batches.
- Simplifies graph tensor handling.

---

#### Step 5: squeeze batch dimension
```python
kyc_x, edge_index = kyc_x[0], edge_index[0]
swift_edge_attr, label = swift_edge_attr[0], label[0]
trade_features = trade_features[0].unsqueeze(0)
```

##### Meaning
Because the DataLoader wraps each item in an outer batch dimension of 1, the code manually removes it.

##### Impact
This is necessary for the current sample shapes, but it is fragile if batch size changes.

##### Important note
This is one of the places that currently needs care because the model expects very specific tensor shapes.

---

#### Step 6: forward pass
```python
logits = self.model(kyc_x, edge_index, swift_edge_attr, seq_data, trade_features)
```

##### What goes in
- sender/receiver KYC graph features
- edge index
- transaction amount
- sequence tensor
- trade features

##### What comes out
- raw logits for 3 classes

---

#### Step 7: loss
```python
loss = self.criterion(logits, label.unsqueeze(0))
```

##### Meaning
Cross-entropy compares predicted logits with true label.

##### Why `unsqueeze(0)`
It makes the label shape compatible with the batch dimension expected by the loss function.

##### Impact
Loss drives gradient updates.

---

#### Step 8: backpropagation
```python
loss.backward()
self.optimizer.step()
```

##### What happens
- `backward()` computes gradients.
- `optimizer.step()` updates model weights.

##### Impact
This is the actual learning step.

---

#### Step 9: return updated weights
```python
return self.get_parameters(config={}), len(self.train_loader.dataset), {}
```

##### Returned values
1. Updated model parameters as NumPy arrays.
2. Number of local samples used.
3. Empty metrics dictionary.

##### Why it matters
The server uses these parameters for aggregation.

---

### 6.6 evaluate
```python
def evaluate(self, parameters, config):
```

#### Purpose
Evaluate the current global model on local test data.

#### Main steps
- load server parameters
- set model to eval mode
- run inference on test samples
- compute loss, accuracy, and macro F1

#### Metrics returned
```python
return float(total_loss / len(self.test_loader)), len(self.test_loader.dataset), {"accuracy": accuracy, "f1_macro": f1_m}
```

##### `total_loss / len(self.test_loader)`
Average test loss.

##### `accuracy`
Fraction of exact correct predictions.

##### `f1_macro`
Macro-averaged F1 across classes.

#### Why F1 macro matters
It treats clean/suspicious/fraud more evenly than accuracy alone.

#### Impact
This is useful when the classes are imbalanced.

---

### 6.7 STR alert check in evaluation
```python
laundering_count = all_preds.count(2)
if laundering_count > 0:
    print(f"\n🚨 STR ALERT [{args.bank.upper()}] - {laundering_count} Typologies detected in evaluation!")
```

#### Meaning
If the model predicts class `2` (fraud), it emits an alert.

#### Impact
This is a simple alert indicator during validation.

---

### 6.8 Main client startup
```python
all_ids = get_all_msg_ids(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS)
dataset = MemgraphTBMLDataset(...)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
```

#### What this means
- Pull all transaction IDs.
- Create the dataset.
- Split into 80% train and 20% test.

#### Why 80/20
This is a common default split for local train/test evaluation.

#### Impact
The model trains on 80% of local samples and evaluates on 20%.

---

### 6.9 DataLoader settings
```python
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
```

#### batch_size=1
- Simplifies graph sample handling.
- Works with the current indexing code.

#### shuffle=True for training
Improves generalization by changing sample order each epoch.

#### shuffle=False for test
Keeps evaluation deterministic.

#### Impact
These settings are safe for this code, though not optimized for large-scale performance.

---

## 7. What the Server Actually Does

### 7.1 Imports
```python
import numpy as numpy
import flwr as flwr
import os
from model import TBML_DetectionModel
```

#### Meaning
- `numpy as numpy` is currently a typo-like alias, but still usable as `numpy.savez(...)`.
- `flwr` is the Flower server package.
- `TBML_DetectionModel` provides the model structure to initialize weights.

#### Impact
The server needs the model only to create the initial parameter shapes.

---

### 7.2 global model path
```python
BASE_DIR = os.path.dirname(__file__)
GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, "global_model.npz")
```

#### Meaning
The aggregated weights are saved in `global_model.npz` under the federated_learning folder.

#### Impact
This file becomes the final global checkpoint used by inference.

---

### 7.3 modelStrategy class
```python
class modelStrategy(flwr.server.strategy.FedAvg):
```

#### Meaning
This custom strategy extends Flower’s built-in FedAvg strategy.

#### Why extend FedAvg
To add custom behavior after aggregation, specifically saving the model at a target round.

---

### 7.4 aggregate_fit
```python
def aggregate_fit(self, curr_round, results, failures):
```

#### Parameters
- `curr_round`: current federated round number
- `results`: client updates received this round
- `failures`: failed clients in this round

#### Purpose
Aggregate client updates using FedAvg and optionally save the global model.

---

#### Step 1: call parent FedAvg
```python
aggregated_weights,_ = super().aggregate_fit(curr_round,results,failures)
```

##### Meaning
Flower averages the model updates from the participating clients.

##### What gets averaged
Each corresponding tensor across clients, such as:
- GAT weights
- GAT biases
- LSTM weights
- LSTM biases
- trade MLP weights
- classifier weights

##### Impact
This creates the next global model.

---

#### Step 2: save at round 10
```python
if aggregated_weights is not None and curr_round==10:
```

##### Meaning
Only save the global model at federated round 10.

##### Range
- Current round target: `10`
- Practical range: any chosen checkpoint round

##### Impact
This acts as the final trained model snapshot.

---

#### Step 3: convert Flower parameters to NumPy arrays
```python
aggregated_array = flwr.common.parameters_to_ndarrays(aggregated_weights)
```

##### What it does
Transforms Flower’s internal parameter object into a Python list of NumPy arrays.

##### Why this is needed
`numpy.savez` expects NumPy-compatible arrays.

##### Impact
This is the bridge between Flower and the saved `.npz` file.

---

#### Step 4: save model weights
```python
numpy.savez(GLOBAL_MODEL_PATH,*aggregated_array)
```

##### Meaning
Writes each tensor array into `global_model.npz`.

##### What is inside the file
A sequence of arrays representing the model tensors in order.

##### Important detail
This save format is order-based, not named-key based.

##### Impact
The inference code must load arrays in the same order the model expects.

---

#### Step 5: return
```python
return aggregated_weights,_
```

##### Meaning
Returns the aggregated parameters and metrics to Flower.

##### Impact
Flower continues the federated loop with the newly averaged weights.

---

### 7.5 Server initialization
```python
model = TBML_DetectionModel(kyc_dim=3,swift_dim=1,trade_dim=3)
```

#### Why server creates a model
Not for training directly, but to get the exact parameter structure and initial tensor shapes.

#### Parameter values
- `kyc_dim=3`
- `swift_dim=1`
- `trade_dim=3`

These match the client sample construction and the model design.

#### Impact
The server and clients must agree on these dimensions exactly.

---

### 7.6 initial_parameters
```python
ndarrays = [val.cpu().numpy() for val in model.state_dict().values()]
initial_parameters = flwr.common.ndarrays_to_parameters(ndarrays)
```

#### What it does
Extracts initial random model weights and wraps them for Flower.

#### Source of values
These come from PyTorch’s default parameter initialization.

#### What are these values
Randomly initialized weight matrices and bias vectors for every learnable layer.

#### Impact
This is the initial global model before any client training begins.

---

### 7.7 server strategy configuration
```python
strategy = modelStrategy(
    fraction_fit =1.0,
    min_fit_clients =3,
    min_available_clients =3,
    initial_parameters = initial_parameters
)
```

#### Parameter details

##### `fraction_fit=1.0`
- Range: 0.0 to 1.0
- Value used: 1.0
- Meaning: use all available clients each round
- Impact: best consistency, slower if clients are down

##### `min_fit_clients=3`
- Range: integer >= 1
- Value used: 3
- Meaning: at least 3 clients must be available for training
- Impact: ensures all banks participate

##### `min_available_clients=3`
- Range: integer >= 1
- Value used: 3
- Meaning: server waits until 3 clients are connected
- Impact: prevents incomplete federation

##### `initial_parameters`
- The starting model weights
- Provided to all clients at the first round

#### Why these values matter
They enforce a strict 3-bank federated setup.

---

### 7.8 start_server
```python
flwr.server.start_server(
    server_address = "0.0.0.0:8085",
    config = flwr.server.ServerConfig(num_rounds=10),
    strategy = strategy
)
```

#### Parameters explained

##### `server_address="0.0.0.0:8085"`
- Means the server listens on all interfaces.
- Port: `8085`
- Impact: clients connect to this port.

##### `num_rounds=10`
- Range: any positive integer
- Value used: 10
- Meaning: run 10 federated training rounds
- Impact: more rounds usually improve convergence but increase runtime

##### `strategy=strategy`
- Uses your custom FedAvg strategy with checkpointing at round 10

---

## 8. What the Global Model Really Is

The global model is not a single mysterious object. It is the same `TBML_DetectionModel` architecture whose parameter tensors have been averaged across clients.

### It contains learnable tensors such as
- GATv2Conv weights and biases
- LSTM weights and biases
- trade branch linear layer weights and biases
- classifier weights and biases

### How it is generated
1. Server starts with initial model weights.
2. Each client trains on its own bank data.
3. Each client sends updated tensors back.
4. Server averages tensors using FedAvg.
5. Averaged tensors become the new global model.
6. At round 10 the final tensors are saved to `global_model.npz`.

### What parameters are coming from where
- Initial parameters: from `TBML_DetectionModel().state_dict()` on the server.
- Client update parameters: from local training on each bank’s graph samples.
- Aggregated parameters: weighted average of client parameters computed by Flower.

---

## 9. Parameter Meaning and Impact

### 9.1 Model dimensions
| Parameter | Value | Meaning | Impact |
|---|---:|---|---|
| `kyc_dim` | 3 | sender/receiver KYC features | Must match feature vector size |
| `swift_dim` | 1 | transaction amount edge feature | Must match edge feature size |
| `trade_dim` | 3 | trade document features | Must match trade feature vector |
| `hidden_dim` | 64 | hidden output size in GAT branch | Higher = more capacity, more cost |
| `lstm_hidden` | 64 | LSTM hidden state size | Controls temporal representation |
| `classes` | 3 | output classes | CLEAN, SUSPICIOUS, FRAUD |

### 9.2 Loss and optimizer
| Parameter | Value | Meaning | Impact |
|---|---:|---|---|
| `CrossEntropyLoss(weight=[0.1, 1.5, 2.0])` | fixed tensor | class imbalance handling | fraud errors matter more |
| `Adam(lr=0.001)` | 0.001 | learning rate | too high destabilizes, too low slows training |
| local epochs | 3 | local steps per round | more epochs = more local learning but slower federation |
| batch size | 1 | one graph sample per batch | easy shape handling, slower training |

### 9.3 Federated settings
| Parameter | Value | Meaning | Impact |
|---|---:|---|---|
| `fraction_fit` | 1.0 | fraction of clients used per round | all clients participate |
| `min_fit_clients` | 3 | minimum clients for fit | requires all banks |
| `min_available_clients` | 3 | required available clients | blocks until all banks are online |
| `num_rounds` | 10 | federated rounds | controls total training cycles |
| save checkpoint round | 10 | final checkpoint round | saves the learned global model |

---

## 10. What Is the Impact of These Parameters

### If you increase hidden dimensions
- Better representation power
- More memory and compute usage
- More overfitting risk

### If you increase local epochs
- Clients adapt more to local bank data
- Can improve local fit
- Can hurt global generalization if too high

### If you increase learning rate
- Faster updates
- Higher chance of unstable training

### If you increase number of rounds
- Better global convergence potential
- Longer training time

### If you change class weights
- Directly affects which misclassifications are penalized most
- Helpful for imbalanced fraud classes

### If you change fraction_fit
- Lower values reduce communication cost
- Higher values use more clients each round
- In this project, 1.0 forces all clients every round

---

## 11. How the Model Parameters Move Through the System

### At initialization
- Server creates initial random weights.
- Flower wraps them as `initial_parameters`.

### At client startup
- Client receives the server parameters.
- Client loads them into its local PyTorch model.

### During local training
- Gradients modify the weights on the client.
- These become the updated client parameters.

### At server aggregation
- Server receives all client parameter lists.
- FedAvg averages each tensor position across clients.

### At final checkpoint
- The averaged tensors are saved into `global_model.npz`.

### During inference
- The inference script loads `global_model.npz`.
- It reconstructs the `state_dict` in the same order.
- The model is used for real-time scoring.

---

## 12. Important Technical Caveats in the Current Code

### 12.1 Order-based serialization
The saved `.npz` uses array order, not parameter names.
That means the inference loader must use exactly the same order as training.

### 12.2 Shape sensitivity
The client dataset returns graph-shaped tensors that must match the model’s expectations.
If the shape changes, training or inference can break.

### 12.3 Default label bias
If `label` is missing from data, the code defaults to `0`.
That can create a strong bias toward CLEAN unless labels are properly added.

### 12.4 One-client-per-bank design
The code is designed for exactly 3 banks.
If you add more banks, the server configuration must change.

---

## 13. Function-by-Function Summary

### client.py
- `get_all_msg_ids`
  - pulls transaction IDs from Memgraph
- `MemgraphTBMLDataset.__getitem__`
  - builds one sample from graph context
- `BankClient.get_parameters`
  - exports model tensors to NumPy
- `BankClient.set_parameters`
  - loads aggregated weights into model
- `BankClient.fit`
  - does local training
- `BankClient.evaluate`
  - computes test metrics and alerts

### server.py
- `modelStrategy.aggregate_fit`
  - FedAvg aggregation + checkpointing
- main block
  - initializes the model and starts the Flower server

### detection.py
- `TBML_DetectionModel.__init__`
  - defines GAT + LSTM + trade MLP + classifier
- `TBML_DetectionModel.forward`
  - produces 3-class logits

---

## 14. End-to-End Step-by-Step Example

1. Start Memgraph for Bank A, B, and C.
2. Start Flower server on port 8085.
3. Start three clients, one per bank.
4. Each client fetches its own transaction IDs.
5. Each client queries sender, receiver, and trade data from its bank graph.
6. Each client converts the graph context to tensors.
7. Each client trains the model locally for 3 epochs.
8. Each client sends its updated parameter tensors to the server.
9. Server averages the tensors using FedAvg.
10. Server repeats for 10 rounds.
11. Server saves final averaged tensors to `global_model.npz`.
12. Inference later loads that file for real-time scoring.

---

## 15. Final Interpretation

In simple terms:
- The **client** is the local bank learner.
- The **server** is the weight aggregator.
- The **global model** is the averaged set of all learnable tensors after multiple federated rounds.
- The **data** comes from each bank’s own graph database, not from shared raw files.
- The **parameters** are the neural network weights and biases from the model layers.
- The **impact** of the parameters depends on architecture size, class weights, learning rate, number of epochs, and number of federated rounds.

The main idea is that the project learns from multiple banks without ever centralizing their raw data.
