import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# 1. LOAD DATA (Crucial to avoid NameError)
kyc = pd.read_csv('kyc.csv')
swift = pd.read_csv('swift.csv')
trade = pd.read_csv('trade_docs.csv')

# 2. RE-ESTABLISH MAPPINGS
ent_map = {id: i for i, id in enumerate(kyc['entity_id'])}
msg_map = {id: i for i, id in enumerate(swift['msg_id'])}

# 3. CREATE NETWORKX GRAPH
G = nx.DiGraph()

# Sampling a very small subset so labels are actually readable
sample_size = 150
sample_txns = set(random.sample(list(msg_map.keys()), min(sample_size, len(msg_map))))

# Add Edges: Entity -> Transaction
for s, m in zip(swift['sender_id'], swift['msg_id']):
    if m in sample_txns and s in ent_map:
        G.add_edge(f"E_{s}", f"T_{m}", label='sends')

# Add Edges: Transaction -> Document
for m, d in zip(trade['swift_msg_id'], trade['doc_id']):
    if m in sample_txns:
        G.add_edge(f"T_{m}", f"D_{d}", label='trade')

# 4. DRAWING WITH LABELS
plt.figure(figsize=(14, 10))

# Spring layout helps space out labels
pos = nx.spring_layout(G, k=0.8, seed=42) 

# Draw Nodes and Node Labels (Entity/Transaction IDs)
nx.draw(G, pos, 
        with_labels=True, 
        node_size=1500, 
        node_color='skyblue', 
        font_size=8, 
        font_weight='bold',
        arrowsize=20)

# Draw Edge Labels ('sends', 'trade')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

plt.title("TBML Trace: Entity to Transaction to Document")
plt.show()