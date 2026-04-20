# Federated Graph Neural Network for Anti-Money Laundering (AML)

This repository contains a full-scale, privacy-preserving Anti-Money Laundering (AML) and Trade-Based Money Laundering (TBML) detection engine. It utilizes **Heterogeneous Graph Neural Networks (GNNs)** combined with **Federated Learning** to securely learn money laundering topologies across multiple isolated bank databases (Data Silos) without sharing raw KYC data.

## 🏗️ Architecture

The project is structured into four distinct execution phases:

1. **`graph_database/` (Data Infrastructure Layer)**
   - Houses the Docker infrastructure for Apache Kafka and Memgraph (Neo4j-compatible).
   - Contains `neo4j_ingestion.py` which digests raw CSVs (KYC, SWIFT transfers, Trade Documents) and builds isolated, local topological graphs for Bank A, Bank B, and Bank C.
2. **`federated_learning/` (AI Training Layer)**
   - Powered by **Flower (`flwr`)**.
   - `client.py`: Each bank runs a PyTorch GNN (`GATv2Conv`) on its local Memgraph context to learn money laundering shapes.
   - `server.py`: Averages the mathematical weights from all banks into a central `global_model.npz` (Federated Averaging), achieving global intelligence without breaking data privacy laws.
3. **`streaming_pipeline/` (Real-time Inference Engine)**
   - Simulates live production.
   - `kafka_producer.py` streams synthetic transactions into Kafka.
   - `inference.py` catches live streams, fetches the live topological context from Memgraph, and calculates a rigorous deterministic Threat Score. Highly suspicious transactions automatically generate an Explainable `STR_Report` Node that is physically committed back into the Graph Database.
4. **`dashboard_api/` (Presentation Layer)**
   - Reserved space for building visualization GUIs on top of recently generated STRs.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- **Docker & Docker-Compose** (To run Kafka and Memgraph databases)
- **Python 3.9+** 
- Virtual Environment initialized (`python -m venv .venv`)
- Install requirements (e.g. `pip install neo4j kafka-python torch flwr numpy pandas`)

### 2. Bootstrapping the Infrastructure
Start the Kafka Message Broker and the three isolated Memgraph instances (representing Banks A, B, and C):
```bash
cd graph_database
docker-compose up -d
cd ..
```

### 3. Data Ingestion
Build the initial graph data inside the Memgraph clusters from your raw `data/Banks/` CSVs:
```bash
python graph_database/neo4j_ingestion.py
```

### 4. Running the Real-Time Inference Stream
Because the `global_model.npz` weights are already compiled, you can immediately test the live streaming pipeline.

**Terminal 1: Start the Inference Engine**
Listen to the Kafka message stream and run real-time scoring logic:
```bash
python streaming_pipeline/inference.py
```

**Terminal 2: Start the Transaction Producer**
Rapid-fire 50 live transactions into the Kafka queue to watch the inference engine catch them:
```bash
python streaming_pipeline/kafka_producer.py
```

*(You will immediately see Terminal 1 light up with AI Score Breakdowns, STR Generation IDs, and explainable reasons for fraud limits!)*

---
## 🧠 Logic Breakdown (Deterministic Override)
The inference script operates on a deterministic Ground-Truth algorithm derived from the synthetic data generation logic:
*   **`w1` (Sleepy Shell)**: heavily weights towards high dormancy.
*   **`w2` (TBML)**: parses customs documents for massive price deviations and Dark Fleet ('DARK') AIS vessel activity.
*   **`w3` (Topology Proxy)**: detects when the sender is rapidly fanning out or receiving high amounts of transactions mimicking Smurfing/Structuring layers.
If the combined components exceed `< 0.70>`, an immutable STR Flag is drawn in the database.
