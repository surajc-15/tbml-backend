import pandas as pd
import numpy as np

def fix_weight_variance(file_path="trade_docs_enriched.csv"):
    df = pd.read_csv(file_path)
    
    # Identify non-fraud rows (the ones stuck at 0.0271)
    # Fraud rows have weight_gap_score == 1.0, leave those alone!
    normal_mask = df['weight_gap_score'] < 0.5 
    
    # Generate unique noise for EVERY row
    # This makes the gap vary between 0.01 and 0.05 (1% to 5% normal variance)
    noise = np.random.uniform(0.95, 0.99, size=normal_mask.sum())
    
    # Re-calculate Actual Weight with high variance
    df.loc[normal_mask, 'actual_weight_kg'] = (df.loc[normal_mask, 'declared_weight_kg'] * noise).round(2)
    
    # Re-calculate Weight Gap Score
    df['weight_gap_score'] = (
        (df['declared_weight_kg'] - df['actual_weight_kg']) / 
        (df['declared_weight_kg'] + 1e-6)
    ).round(4)
    
    df.to_csv(file_path, index=False)
    print(f"✅ Fixed variance in {file_path}. No more 0.0271 constants!")

# Run for all bank folders
for bank in ['bank_a_data', 'bank_b_data', 'bank_c_data']:
    path = f"{bank}/trade_docs_enriched.csv"
    fix_weight_variance(path)