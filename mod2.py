import pandas as pd
import numpy as np
import os

# Define the folders where your bank data is stored
BANKS = ['bank_a_data', 'bank_b_data', 'bank_c_data']

def enrich_bank_data():
    for bank in BANKS:
        trade_path = os.path.join(bank, "trade_docs.csv")
        
        if not os.path.exists(trade_path):
            print(f"⚠️ Skipping {bank}: trade_docs.csv not found.")
            continue
            
        print(f"🏗️  Processing {bank}...")
        df = pd.read_csv(trade_path)

        # 1. Generate Physical Weights (Phantom Shipping base)
        # Assuming qty exists based on your headers
        df['declared_weight_kg'] = df['qty'] * 12.5 
        df['actual_weight_kg'] = (df['declared_weight_kg'] * np.random.uniform(0.97, 1.03)).round(2)
        
        # 2. Logistics Indicators
        df['ais_status'] = "Active"
        df['port_log_status'] = "Verified"

        # 3. Identify Fraudulent Rows for Phantom Injection
        # We use your 'price_deviation' header as the anchor
        fraud_mask = df['price_deviation'] > 55

        # --- THE PHANTOM INJECTION ---
        df.loc[fraud_mask, 'actual_weight_kg'] = 0.0
        df.loc[fraud_mask, 'ais_status'] = "Disconnected"
        df.loc[fraud_mask, 'port_log_status'] = "No_Record"
        df.loc[fraud_mask, 'vessel_id'] = "GHOST_VESSEL_X"

        # 4. Feature Engineering (The GNN inputs)
        # Weight Gap Score (1.0 = 100% missing cargo)
        df['weight_gap_score'] = (
            (df['declared_weight_kg'] - df['actual_weight_kg']) / 
            (df['declared_weight_kg'] + 1e-6)
        ).round(4)

        # Device Entropy (Cyber risk)
        df['device_entropy'] = np.random.uniform(0.1, 0.25, size=len(df))
        df.loc[fraud_mask, 'device_entropy'] = np.random.uniform(0.75, 0.99)

        # 5. Save as 'trade_docs_enriched.csv' in the SAME bank folder
        output_path = os.path.join(bank, "trade_docs_enriched.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Enriched {bank} data. Saved to {output_path}")

if __name__ == "__main__":
    enrich_bank_data()