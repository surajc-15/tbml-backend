import pandas as pd
import numpy as np
import random

# ================================
# 1. LOAD SYNTHETIC DATA
# ================================
bank = pd.read_csv('synthetic_bank_data.csv')
trade = pd.read_csv('synthetic_trade_data.csv')
device = pd.read_csv('synthetic_device_data.csv')

# ================================
# 2. SAFE MERGE (FIXED AMOUNT ISSUE)
# ================================
# Remove duplicate 'amount' from trade before merge
trade_clean = trade.drop(columns=['amount'], errors='ignore')

df = bank.merge(trade_clean, on='txn_id').merge(
    device, on=['txn_id', 'account_id']
)

# Debug (optional)
# print(df.columns)

# ================================
# 3. MARKET PRICE MAPPING
# ================================
market_price_map = {
    "Ready-made Garments": 8.5,
    "Raw Jute": 5.0,
    "Smartphones": 300,
    "Solar Panels": 120,
    "Industrial Chemicals": 50
}

# ================================
# 4. RECEIVER ENTITY POOL
# ================================
receiver_pool = [f"ENT_UK_{i:03d}" for i in range(1, 20)]

# ================================
# 5. CREATE KYC DATASET (NODES)
# ================================
kyc_final = pd.DataFrame({
    'entity_id': df['account_id'],
    'jurisdiction': np.random.choice(['Bangladesh', 'India', 'UAE'], len(df)),
    'incorp_date': np.random.choice(['2020-01-01', '2021-06-15', '2022-09-10'], len(df)),
    'is_shell': df['is_dormant_flag'],
    'dorm_days': [
        random.randint(500, 1500) if x == 1 else random.randint(1, 50)
        for x in df['is_dormant_flag']
    ],
    'device_hash': df['device_id_hash'],
    'industry': np.random.choice(['Textiles', 'Electronics', 'Chemicals'], len(df))
}).drop_duplicates(subset='entity_id')

# ================================
# 6. CREATE SWIFT DATA (FINANCIAL EDGES)
# ================================
swift_final = pd.DataFrame({
    'msg_id': df['txn_id'],
    'sender_id': df['account_id'],
    'receiver_id': np.random.choice(receiver_pool, len(df)),
    'amount': df['amount'],   # ✅ FIXED: now exists correctly
    'currency': 'USD',
    'timestamp': df['timestamp'],
    'device_ip': df['ip_address']
})

# ================================
# 7. CREATE TRADE DOCS (TRADE EDGES)
# ================================
trade_final = pd.DataFrame({
    'doc_id': df['ship_id'],
    'swift_msg_id': df['txn_id'],
    'commodity': df['commodity'],
    'unit_price': df['unit_price'],
    'market_avg': df['commodity'].map(market_price_map),
    'qty': df['quantity'],
    'vessel_id': df['ship_id']
})

# ================================
# 8. ADD TBML RISK FEATURE
# ================================
trade_final['price_deviation'] = (
    trade_final['unit_price'] - trade_final['market_avg']
) / trade_final['market_avg']

# ================================
# 9. SAVE OUTPUT FILES
# ================================
kyc_final.to_csv('kyc.csv', index=False)
swift_final.to_csv('swift.csv', index=False)
trade_final.to_csv('trade_docs.csv', index=False)

# ================================
# 10. SUMMARY
# ================================
print("\n✅ SUCCESS: TBML Schema Generated")
print(f"KYC records: {len(kyc_final)}")
print(f"SWIFT records: {len(swift_final)}")
print(f"TRADE records: {len(trade_final)}")

print("\n📂 Files created:")
print(" - kyc.csv (Entities / Nodes)")
print(" - swift.csv (Financial Transactions)")
print(" - trade_docs.csv (Trade Documents)")