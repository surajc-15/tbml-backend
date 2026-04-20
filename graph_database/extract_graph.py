import torch
from neo4j import GraphDatabase
import pandas as pd

# Connect to Bank C (Running on port 7689 on your Mac)
URI = "bolt://127.0.0.1:7689"
AUTH = ("", "")

def build_pytorch_graph():
    print("🔌 Connecting to Bank C Memgraph...")
    driver = GraphDatabase.driver(URI, auth=AUTH)

    with driver.session() as session:
        # 1. Fetch all Accounts
        print("👤 Fetching Account Nodes...")
        nodes_res = session.run("MATCH (a:Account) RETURN a.id AS id")
        nodes_data = [record.data() for record in nodes_res]
        
        # 2. Fetch all Transaction Paths (Account -> Transaction -> Account)
        print("🕸️ Fetching Money Flow Edges...")
        edges_res = session.run("""
            MATCH (s:Account)-[:SENDS]->(t:Transaction)-[:TO]->(r:Account) 
            RETURN s.id AS src, r.id AS dst
        """)
        edges_data = [record.data() for record in edges_res]

    driver.close()

    print("\n⚙️ Converting to PyTorch Tensors...")
    # Create the PyTorch ID Mapping (String ID -> Integer 0 to N-1)
    node_mapping = {row['id']: i for i, row in enumerate(nodes_data)}

    # Build the edge_index array
    src_nodes = []
    dst_nodes = []
    
    for row in edges_data:
        # Only add the edge if both accounts exist in our mapping
        if row['src'] in node_mapping and row['dst'] in node_mapping:
            src_nodes.append(node_mapping[row['src']])
            dst_nodes.append(node_mapping[row['dst']])

    # Create the official PyTorch Tensor
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    print(f"\n✅ PyTorch Graph Successfully Built!")
    print(f"📊 Total Nodes Mapped: {len(node_mapping)}")
    print(f"📊 Edge Index Shape:   {edge_index.shape}")
    
    return edge_index, node_mapping

if __name__ == "__main__":
    edge_index, mapping = build_pytorch_graph()