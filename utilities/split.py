import pandas as pd
import os
import numpy as np

from utilities.path_config import get_bank_file, get_bank_path

banks = ['banka', 'bankb', 'bankc']

for bank in banks:
    os.makedirs(get_bank_path(bank), exist_ok=True)

kyc = pd.read_csv('kyc.csv')
swift = pd.read_csv('swift.csv')
trade = pd.read_csv('trade_docs.csv')

# Non-IID split (realistic)
kyc = kyc.sample(frac=1, random_state=42).reset_index(drop=True)

splits = [
    kyc.iloc[:int(0.5*len(kyc))],
    kyc.iloc[int(0.5*len(kyc)):int(0.8*len(kyc))],
    kyc.iloc[int(0.8*len(kyc)):]
]

kyc_dict = dict(zip(banks, splits))

for name, kyc_df in kyc_dict.items():

    # SWIFT for this bank
    bank_swift = swift[swift['sender_id'].isin(kyc_df['entity_id'])]

    # Trade docs
    bank_trade = trade[trade['swift_msg_id'].isin(bank_swift['msg_id'])]

    # Include receiver entities
    receiver_entities = set(bank_swift['receiver_id'].unique())
    combined_entities = set(kyc_df['entity_id']).union(receiver_entities)

    bank_kyc = kyc[kyc['entity_id'].isin(combined_entities)]

    # Reset index
    bank_swift = bank_swift.reset_index(drop=True)
    bank_trade = bank_trade.reset_index(drop=True)
    bank_kyc = bank_kyc.reset_index(drop=True)

    # Save
    bank_kyc.to_csv(get_bank_file(name, 'kyc.csv'), index=False)
    bank_swift.to_csv(get_bank_file(name, 'swift.csv'), index=False)
    bank_trade.to_csv(get_bank_file(name, 'trade_docs.csv'), index=False)

    print(f"✅ {name.upper()} Created:")
    print(f" Entities: {len(bank_kyc)}")
    print(f" Transactions: {len(bank_swift)}")
    print(f" Trade Docs: {len(bank_trade)}\n")

print("🚀 All silos ready for Federated Learning!")