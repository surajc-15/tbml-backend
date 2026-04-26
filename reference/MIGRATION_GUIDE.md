# TBML Detection System - Implementation Guide & Quick Start

## Summary
This guide explains how to migrate the repo to the improved TBML setup.

## Main Changes
- Fix the server import typo.
- Add cross-modal attention to the model.
- Fix tensor handling and class weights in the client.
- Replace the old inference script with `inference_v2.py`.
- Add transaction labels, validation, and logging in ingestion.

## Recommended Workflow
1. Start Kafka and graph services.
2. Run ingestion.
3. Train federated clients.
4. Start real-time inference.
5. Stream test transactions.

## Important Notes
- The old global checkpoint is not compatible with the revised architecture.
- Retraining is recommended after the model changes.
- STR output should include confidence, heuristic breakdown, and reason text.

## Troubleshooting
- Missing model file: retrain first.
- Parameter mismatch: expected after architecture changes.
- Memgraph unavailable: start the containers.
- No STRs created: verify graph context and Kafka flow.
