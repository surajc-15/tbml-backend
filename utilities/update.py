import pandas as pd
import numpy as np

from utilities.path_config import get_bank_file

def update_to_tbml(bank_prefix):
    # Load existing files
    kyc = pd.read_csv(get_bank_file(bank_prefix, "kyc.csv"))
    swift = pd.read_csv(get_bank_file(bank_prefix, "swift.csv"))
    trade = pd.read_csv(get_bank_file(bank_prefix, "trade_docs.csv"))

    # 1. Feature: Device Entropy (Count unique IPs per entity)
    entropy = swift.groupby('sender_id')['device_ip'].nunique().reset_index()
    entropy.columns = ['entity_id', 'device_entropy']
    kyc = kyc.merge(entropy, on='entity_id', how='left').fillna(1)

    # 2. Logic: Define Transaction Labels
    # Join everything to SWIFT to evaluate each transaction
    m = swift.merge(kyc[['entity_id', 'is_shell', 'dorm_days', 'device_entropy']], left_on='sender_id', right_on='entity_id', how='left')
    m = m.merge(trade[['swift_msg_id', 'price_deviation']], left_on='msg_id', right_on='swift_msg_id', how='left')

    def classify(row):
        # 2 = Laundering (STR Trigger)
        if row['price_deviation'] > 150: return 2  # Extreme Over-invoicing
        if pd.isna(row['swift_msg_id']): return 2  # Phantom Shipping (No trade doc)
        if row['is_shell'] == 1: return 2         # Known Shell entity
        
        # 1 = Watchlist
        if row['device_entropy'] > 5: return 1    # High Device Entropy
        if row['dorm_days'] > 500: return 1       # High Dormancy
        
        # 0 = Legitimate
        return 0

    swift['label'] = m.apply(classify, axis=1)
    
    # Save back updated files
    kyc.to_csv(get_bank_file(bank_prefix, "kyc.csv"), index=False)
    swift.to_csv(get_bank_file(bank_prefix, "swift.csv"), index=False)
    print(f"✅ Updated {bank_prefix} with TBML labels.")

for b in ['banka', 'bankb', 'bankc']:
    update_to_tbml(b)