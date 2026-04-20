import pandas as pd
import numpy as np
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer

# ================================
# 1. LOAD DATA
# ================================
bank = pd.read_csv("seed_bank.csv")
trade = pd.read_csv("seed_trade.csv")
device = pd.read_csv("seed_device.csv")

data = {
    "bank": bank,
    "trade": trade,
    "device": device
}

# ================================
# 2. AUTO-DETECT METADATA
# ================================
metadata = MultiTableMetadata()
metadata.detect_from_dataframes(data)

# ================================
# 3. FIX IMPORTANT TYPES ONLY
# ================================

# IDs (important for graph + uniqueness)
metadata.update_column('bank', 'txn_id', sdtype='id')
metadata.update_column('bank', 'account_id', sdtype='id')

metadata.update_column('trade', 'txn_id', sdtype='id')
metadata.update_column('trade', 'ship_id', sdtype='id')

metadata.update_column('device', 'txn_id', sdtype='id')
metadata.update_column('device', 'account_id', sdtype='id')
metadata.update_column('device', 'device_id_hash', sdtype='id')

# Datetime
metadata.update_column('bank', 'timestamp', sdtype='datetime')

# ================================
# 4. TRAIN MODEL
# ================================
print("🚀 Training HMA Synthesizer...")

synthesizer = HMASynthesizer(metadata, verbose=True)
synthesizer.fit(data)

# ================================
# 5. GENERATE DATA
# ================================
num_rows = 100000
scale = num_rows / len(bank)

print(f"🧪 Generating {num_rows} rows...")

synthetic_data = synthesizer.sample(scale=scale)

bank_syn = synthetic_data['bank']
trade_syn = synthetic_data['trade']
device_syn = synthetic_data['device']

# ================================
# 6. POST-PROCESSING
# ================================

# Fix amount consistency
if 'unit_price' in trade_syn.columns and 'quantity' in trade_syn.columns:
    trade_syn['amount'] = trade_syn['unit_price'] * trade_syn['quantity']

# Remove duplicates
bank_syn = bank_syn.drop_duplicates(subset='txn_id')
trade_syn = trade_syn.drop_duplicates(subset='txn_id')
device_syn = device_syn.drop_duplicates(subset='txn_id')

# Optional realism noise
if 'amount' in bank_syn.columns:
    bank_syn['amount'] *= np.random.uniform(0.98, 1.02, len(bank_syn))

# ================================
# 7. SAVE
# ================================
bank_syn.to_csv("synthetic_bank_data.csv", index=False)
trade_syn.to_csv("synthetic_trade_data.csv", index=False)
device_syn.to_csv("synthetic_device_data.csv", index=False)

print("\n✅ SUCCESS! Synthetic AML dataset generated.")