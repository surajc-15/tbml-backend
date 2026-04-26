import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class TBML_DetectionModel(nn.Module):
    def __init__(self, kyc_dim=3, swift_dim=1, trade_dim=3, hidden_dim=64, lstm_hidden=64, classes=3):
        super(TBML_DetectionModel, self).__init__()
        
        # Branch A: Graph + Temporal
        self.gat = GATv2Conv(in_channels=kyc_dim, out_channels=hidden_dim, edge_dim=swift_dim, heads=4, concat=False)
        self.lstm = nn.LSTM(input_size=hidden_dim + swift_dim, hidden_size=lstm_hidden, batch_first=True)

        # Branch B: Trade MLP (Added BatchNorm for training stability)
        self.trade_mlp = nn.Sequential(
            nn.Linear(trade_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # Fusion Layer
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, classes)
        )

    def forward(self, kyc_x, edge_index, swift_edge_attr, seq_data, trade_features):
        # 1. Process Graph branch (GAT)
        enriched_nodes = F.relu(self.gat(kyc_x, edge_index, edge_attr=swift_edge_attr))
        
        # --- THE FIX: PRIMARY ACTOR POOLING ---
        # Instead of averaging the whole graph and diluting the risk, 
        # we average ONLY Node 0 (Sender) and Node 1 (Receiver). 
        # Because of GAT message passing, these two nodes already contain the risk of their neighbors!
        sender_node = enriched_nodes[0]
        receiver_node = enriched_nodes[1]
        graph_context = torch.stack([sender_node, receiver_node]).mean(dim=0).reshape(1, 1, -1) 
        
        # 2. Fix the shape for LSTM
        if seq_data.dim() == 4:
            seq_data = seq_data.squeeze(0) 

        # 3. FUSION: Combine the graph context with the raw sequence data
        graph_context_expanded = graph_context.expand(seq_data.size(0), seq_data.size(1), -1)
        combined_seq = torch.cat([seq_data, graph_context_expanded], dim=-1)

        # 4. Feed the combined vector to LSTM
        lstm_out, _ = self.lstm(combined_seq)
        final_lstm_step = lstm_out[:, -1, :] 
        
        # 5. Process Trade branch
        trade_emb = self.trade_mlp(trade_features)
        
        # 6. Final Fusion
        fused_vector = torch.cat([final_lstm_step, trade_emb], dim=1)
        return self.classifier(fused_vector)