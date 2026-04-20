$ErrorActionPreference = "Stop"

# Create Base Directories
New-Item -ItemType Directory -Force -Path "1_graph_database\data"
New-Item -ItemType Directory -Force -Path "2_federated_learning"
New-Item -ItemType Directory -Force -Path "3_streaming_pipeline"
New-Item -ItemType Directory -Force -Path "4_dashboard_api"

# Phase 1: Database & Graph Construction
Move-Item -Path "docker-compose.yml" -Destination "1_graph_database\" -ErrorAction SilentlyContinue
Move-Item -Path "neo4j_ingestion.py" -Destination "1_graph_database\" -ErrorAction SilentlyContinue
Move-Item -Path "extract_graph.py" -Destination "1_graph_database\" -ErrorAction SilentlyContinue
Move-Item -Path "graph.py" -Destination "1_graph_database\" -ErrorAction SilentlyContinue
Move-Item -Path "graph_master.py" -Destination "1_graph_database\" -ErrorAction SilentlyContinue

# Move Data CSVs and folders out of root into 1_graph_database/data/
Move-Item -Path "Banks" -Destination "1_graph_database\data\" -ErrorAction SilentlyContinue
Move-Item -Path "bank_a_data" -Destination "1_graph_database\data\" -ErrorAction SilentlyContinue
Move-Item -Path "bank_b_data" -Destination "1_graph_database\data\" -ErrorAction SilentlyContinue
Move-Item -Path "bank_c_data" -Destination "1_graph_database\data\" -ErrorAction SilentlyContinue
Move-Item -Path "*.csv" -Destination "1_graph_database\data\" -ErrorAction SilentlyContinue

# Phase 2: Federated Learning
Move-Item -Path "client.py" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue
Move-Item -Path "server.py" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue
Move-Item -Path "client2.ipynb" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue
Move-Item -Path "server.ipynb" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue
Move-Item -Path "global_model.npz" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue
Move-Item -Path "tbml" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue
Move-Item -Path "model.py" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue
Move-Item -Path "mod*.py" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue
Move-Item -Path "*.png" -Destination "2_federated_learning\" -ErrorAction SilentlyContinue

# Phase 3: Streaming
Move-Item -Path "kafka_pipeline.py" -Destination "3_streaming_pipeline\" -ErrorAction SilentlyContinue
Move-Item -Path "kafka_producer.py" -Destination "3_streaming_pipeline\" -ErrorAction SilentlyContinue
Move-Item -Path "inference.py" -Destination "3_streaming_pipeline\" -ErrorAction SilentlyContinue

# Utilities / Miscs
New-Item -ItemType Directory -Force -Path "utilities" -ErrorAction SilentlyContinue
Move-Item -Path "gen.py" -Destination "utilities\" -ErrorAction SilentlyContinue
Move-Item -Path "seed.py" -Destination "utilities\" -ErrorAction SilentlyContinue
Move-Item -Path "split.py" -Destination "utilities\" -ErrorAction SilentlyContinue
Move-Item -Path "trigger.py" -Destination "utilities\" -ErrorAction SilentlyContinue
Move-Item -Path "update.py" -Destination "utilities\" -ErrorAction SilentlyContinue
Move-Item -Path "visualize.py" -Destination "utilities\" -ErrorAction SilentlyContinue

Write-Host "Restructuring Complete!"
