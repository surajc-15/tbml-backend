import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

# LOAD
kyc = pd.read_csv('kyc.csv')
swift = pd.read_csv('swift.csv')
trade = pd.read_csv('trade_docs.csv')

data = HeteroData()

# ENTITY NODES
le_ind = LabelEncoder()
le_jur = LabelEncoder()

data['entity'].x = torch.tensor(list(zip(
    kyc['is_shell'],
    kyc['dorm_days'],
    le_ind.fit_transform(kyc['industry']),
    le_jur.fit_transform(kyc['jurisdiction'])
)), dtype=torch.float)

# TRANSACTION NODES
data['transaction'].x = torch.tensor(
    swift['amount'].values, dtype=torch.float
).view(-1, 1)

# DOCUMENT NODES
trade['delta'] = trade['unit_price'] - trade['market_avg']

data['document'].x = torch.tensor(
    trade[['unit_price', 'market_avg', 'delta']].values,
    dtype=torch.float
)

# MAPPINGS
ent_map = {id: i for i, id in enumerate(kyc['entity_id'])}
msg_map = {id: i for i, id in enumerate(swift['msg_id'])}
doc_map = {id: i for i, id in enumerate(trade['doc_id'])}

# ENTITY → TRANSACTION
edges = [
    (ent_map[s], msg_map[m])
    for s, m in zip(swift['sender_id'], swift['msg_id'])
    if s in ent_map and m in msg_map
]

data['entity', 'sends', 'transaction'].edge_index = torch.tensor(edges, dtype=torch.long).t()

# TRANSACTION → DOCUMENT
edges = [
    (msg_map[m], doc_map[d])
    for m, d in zip(trade['swift_msg_id'], trade['doc_id'])
    if m in msg_map and d in doc_map
]

data['transaction', 'linked_to', 'document'].edge_index = torch.tensor(edges, dtype=torch.long).t()

# ADD RECEIVER RELATIONSHIP
receiver_ids = swift['receiver_id'].unique()

for r in receiver_ids:
    if r not in ent_map:
        ent_map[r] = len(ent_map)

# Expand entity features
num_new = len(ent_map) - data['entity'].x.shape[0]
if num_new > 0:
    data['entity'].x = torch.cat([
        data['entity'].x,
        torch.zeros((num_new, data['entity'].x.shape[1]))
    ], dim=0)

edges = [
    (msg_map[m], ent_map[r])
    for m, r in zip(swift['msg_id'], swift['receiver_id'])
    if m in msg_map and r in ent_map
]

data['transaction', 'received_by', 'entity'].edge_index = torch.tensor(edges, dtype=torch.long).t()

print("🚀 Heterogeneous Graph Created!")
print(data)