# Major Project — Trade‑Based AML (TBML) with Memgraph + Kafka + Federated Learning

This repository implements a **trade‑based anti‑money‑laundering detection system** using:

- **Memgraph** as the graph database (one instance per bank)
- **Flower (FLWR)** for **federated learning** across banks
- **Kafka** for **event streaming** (real‑time transaction events)
- A **PyTorch / PyTorch Geometric** model that fuses:
  - KYC/account signals (node features)
  - SWIFT/payment signals (edge + temporal signals)
  - Trade document signals (document features)

## High-level flow

1. Each bank has its own Memgraph instance (ports `4000`, `4001`, `4002`).
2. Bank data is ingested into Memgraph (`Account → Transaction → Account` and `Transaction → Document`).
3. Each bank runs a federated client to train locally on its own graph.
4. The central federated server aggregates weights and saves `global_model.npz`.
5. A real‑time inference service consumes Kafka events (topic `incoming_swift_messages`), fetches graph context from Memgraph, and predicts `CLEAN / REVIEW / STR`.

---

## Prerequisites

- Linux (tested) / macOS
- Python 3.9+
- Docker + Docker Compose

Python packages used in code:
- `flwr`, `torch`, `torch-geometric`, `pandas`, `numpy`, `neo4j` (Bolt driver), `kafka-python`, `scikit-learn`

> Note: Installing **PyTorch Geometric** depends on your CUDA/CPU environment. Install PyTorch first, then follow the official PyG install instructions for your platform.

---

## 1) Start Kafka

This repo includes [docker-compose.yml](docker-compose.yml) which exposes Kafka on **`localhost:19092`**.

1. Create the docker network (only needed once):

```bash
docker network create major_project_network
```

2. Start Kafka:

```bash
docker compose up -d
```

Kafka broker address used by the Python scripts:
- default: `localhost:19092`
- override (optional):

```bash
export KAFKA_BROKER=localhost:19092
```

---

## 2) Start Memgraph (3 banks)

Run **three** Memgraph instances locally (one per bank), each exposing Bolt on ports `4000`, `4001`, `4002`.

Example using `memgraph/memgraph-platform` (includes Memgraph + UI):

```bash
# Bank A
docker run -d --name memgraph_banka \
  -p 4000:7687 -p 3000:3000 \
  memgraph/memgraph-platform

# Bank B
docker run -d --name memgraph_bankb \
  -p 4001:7687 -p 3001:3000 \
  memgraph/memgraph-platform

# Bank C
docker run -d --name memgraph_bankc \
  -p 4002:7687 -p 3002:3000 \
  memgraph/memgraph-platform
```

(Optional) Memgraph UI:
- Bank A: http://localhost:3000
- Bank B: http://localhost:3001
- Bank C: http://localhost:3002

---

## 3) Provide bank datasets (local only)

The large datasets folder `Banks/` is **ignored** by git (see [.gitignore](.gitignore)).

Expected layout:

```text
Banks/
  Bank_A/
    kyc.csv
    swift.csv
    trade.csv
  Bank_B/
    kyc.csv
    swift.csv
    trade.csv
  Bank_C/
    kyc.csv
    swift.csv
    trade.csv
```

Place your downloaded datasets into that structure.

---

## 4) Ingest data into Memgraph

Ingest all 3 banks into their respective Memgraph instances:

```bash
python neo4j_ingestion.py
```

If your `Banks/` folder is somewhere else, point ingestion to it:

```bash
export BANKS_DIR=/absolute/path/to/Banks
python neo4j_ingestion.py
```

---

## 5) Federated training (Flower)

1. Start the federated server:

```bash
python server.py
```

2. In **three separate terminals**, start one client per bank:

```bash
python client.py --bank banka
python client.py --bank bankb
python client.py --bank bankc
```

After training finishes, the server saves the aggregated model weights to:
- `global_model.npz`

---

## 6) Real-time inference with Kafka

1. Start the inference consumer (uses Memgraph Bank A for graph context by default):

```bash
python inference.py
```

2. Send a transaction event into Kafka (publishes to topic `incoming_swift_messages`):

```bash
python trigger.py
```

The inference service will:
- read the Kafka message
- fetch sender/receiver/document context from Memgraph
- run the model and print:
  - `CLEAN` (0)
  - `REVIEW` (1)
  - `STR ALERT` (2)

---

## Notes / Troubleshooting

### Kafka port
- This repo’s Kafka compose maps container `9092` → host `19092`.
- Scripts default to `localhost:19092` and can be overridden with `KAFKA_BROKER`.

### Memgraph ports
- The federated clients expect Bolt at:
  - Bank A: `bolt://localhost:4000`
  - Bank B: `bolt://localhost:4001`
  - Bank C: `bolt://localhost:4002`

### Ignored large files
- `Banks/` is intentionally not pushed to GitHub because GitHub blocks files over 100MB.

---

## Key files

- Federated server: [server.py](server.py)
- Federated client: [client.py](client.py)
- Model definition: [model.py](model.py)
- Memgraph ingestion: [neo4j_ingestion.py](neo4j_ingestion.py)
- Real-time inference: [inference.py](inference.py)
- Kafka trigger (event producer): [trigger.py](trigger.py)

(Older demo scripts using Neo4j are still present but are not required for the Memgraph-only pipeline.)
