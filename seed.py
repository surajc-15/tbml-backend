import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_full_master_seed(n=1000):
    # --- Configuration ---
    banks = ['BANK_BD_01', 'BANK_BD_02', 'BANK_UK_01', 'BANK_UK_02']
    commodities = ['Ready-made Garments', 'Raw Jute', 'Smartphones', 'Solar Panels', 'Industrial Chemicals']
    ports = ['Chittagong', 'Dhaka', 'Port of London', 'Southampton', 'Singapore']

    # Force graph structure (collisions)
    account_pool = [f"ACC_{i:03d}" for i in range(100, 300)]
    device_pool = [f"DEV_HASH_{i:03d}" for i in range(1, 50)]

    # HS Code mapping (realistic)
    hs_map = {
        "Ready-made Garments": "610910",
        "Raw Jute": "530310",
        "Smartphones": "851713",
        "Solar Panels": "854143",
        "Industrial Chemicals": "382499"
    }

    rows = []

    for i in range(n):
        txn_id = f"TXN_{5000+i}"
        acc_id = random.choice(account_pool)
        bank_id = random.choice(banks)

        # --- Fraud decision ---
        is_fraud = random.random() < 0.12

        # --- FRAUD SCENARIOS ---
        if is_fraud:
            fraud_type = random.choice([
                "OVER_INVOICING",
                "UNDER_INVOICING",
                "DEVICE_SHARING"
            ])

            commodity = random.choice(["Industrial Chemicals", "Smartphones"])

            # Pattern 1: Over-invoicing
            if fraud_type == "OVER_INVOICING":
                u_price = random.randint(2000, 5000)
                qty = random.randint(500, 1000)

            # Pattern 2: Under-invoicing
            elif fraud_type == "UNDER_INVOICING":
                u_price = random.randint(1, 5)
                qty = random.randint(500, 2000)

            # Pattern 3: Device sharing fraud
            else:
                u_price = random.randint(1000, 3000)
                qty = random.randint(200, 800)

            amount = u_price * qty

            # Suspicious behavior
            is_dormant = 1
            dev_hash = random.choice(device_pool[:5])  # shared suspicious pool
            vpn_flag = 1
            entropy = random.uniform(0.7, 0.98)

        # --- NORMAL TRANSACTIONS ---
        else:
            fraud_type = "NORMAL"
            commodity = random.choice(commodities)
            u_price = random.randint(15, 60)
            qty = random.randint(100, 2000)
            amount = u_price * qty

            is_dormant = 0
            dev_hash = random.choice(device_pool)
            vpn_flag = 0
            entropy = random.uniform(0.01, 0.25)

        # --- HS Code ---
        hs_code = hs_map[commodity]

        # --- Timestamp (temporal realism) ---
        base_date = datetime(2026, 1, 1)
        random_days = random.randint(0, 90)
        timestamp = base_date + timedelta(
            days=random_days,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        # --- IP Address ---
        ip_address = f"103.{random.randint(1,250)}.{random.randint(1,250)}.{random.randint(1,250)}"

        # --- Append ---
        rows.append({
            # Core
            'txn_id': txn_id,
            'account_id': acc_id,
            'bank_id': bank_id,
            'amount': amount,
            'currency': 'USD',
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'is_dormant_flag': is_dormant,

            # Trade
            'ship_id': f"SHIP_V_{1000+i}",
            'hs_code': hs_code,
            'commodity': commodity,
            'unit_price': u_price,
            'quantity': qty,
            'origin_port': random.choice(ports),

            # Device
            'device_id_hash': dev_hash,
            'ip_address': ip_address,
            'proxy_vpn_flag': vpn_flag,
            'login_entropy_score': entropy,

            # Labels
            'label': 1 if is_fraud else 0,
            'fraud_type': fraud_type
        })

    return pd.DataFrame(rows)


# --- Generate Data ---
master_seed = generate_full_master_seed(1000)

# --- Split into Silos ---
bank_silo = master_seed[
    ['txn_id', 'account_id', 'bank_id', 'amount', 'currency', 'timestamp', 'is_dormant_flag', 'label']
]

trade_silo = master_seed[
    ['txn_id', 'ship_id', 'hs_code', 'commodity', 'unit_price', 'quantity', 'amount', 'origin_port']
]

device_silo = master_seed[
    ['txn_id', 'account_id', 'device_id_hash', 'ip_address', 'proxy_vpn_flag', 'login_entropy_score']
]

# --- Save ---
bank_silo.to_csv('seed_bank.csv', index=False)
trade_silo.to_csv('seed_trade.csv', index=False)
device_silo.to_csv('seed_device.csv', index=False)
master_seed.to_csv('master_training_seed.csv', index=False)

print("✅ SUCCESS: Generated realistic multi-silo AML dataset (1000 rows)")
print("Files:")
print(" - seed_bank.csv")
print(" - seed_trade.csv")
print(" - seed_device.csv")
print(" - master_training_seed.csv")