import pandas as pd
import os
from neo4j import GraphDatabase

# --- CONFIG ---
# Default expects bank datasets checked out at ./Banks/Bank_A, ./Banks/Bank_B, ./Banks/Bank_C
# Override with BANKS_DIR if your data lives elsewhere.
BASE_BANKS_DIR = os.getenv("BANKS_DIR", os.path.join(os.path.dirname(__file__), "Banks"))

BANKS = {
    "banka": {
        "uri": "bolt://localhost:4000",
        "path": os.path.join(BASE_BANKS_DIR, "Bank_A"),
    },
    "bankb": {
        "uri": "bolt://localhost:4001",
        "path": os.path.join(BASE_BANKS_DIR, "Bank_B"),
    },
    "bankc": {
        "uri": "bolt://localhost:4002",
        "path": os.path.join(BASE_BANKS_DIR, "Bank_C"),
    },
}
CHUNK_SIZE = 50000

def ingest_bank(bank_name, config):
    print(f"\n🚀 Starting {bank_name}...")
    driver = GraphDatabase.driver(config['uri'], auth=("", ""))
    path = config['path']
    
    with driver.session() as session:
        session.run("CREATE INDEX ON :Account(id)")
        session.run("CREATE INDEX ON :Transaction(id)")
        session.run("CREATE INDEX ON :Document(id)")
        session.run("CREATE INDEX ON :Document(swift_ref)")

    # 1. Accounts (NOW WITH KYC ML FEATURES)
    print(f"   👤 Loading Accounts for {bank_name}...")
    for chunk in pd.read_csv(os.path.join(path, "kyc.csv"), chunksize=CHUNK_SIZE):
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS row 
                MERGE (a:Account {id: row.entity_id}) 
                SET a.jurisdiction = row.jurisdiction,
                    a.is_shell = toFloat(row.is_shell),
                    a.dorm_days = toFloat(row.dorm_days),
                    a.device_entropy = toFloat(row.device_entropy)
            """, batch=chunk.to_dict('records'))
    
    # 2. Documents (NOW WITH TBML ML FEATURES)
    print(f"   📄 Loading Documents for {bank_name}...")
    for chunk in pd.read_csv(os.path.join(path, "trade.csv"), chunksize=CHUNK_SIZE):
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS row 
                CREATE (d:Document {id: row.doc_id}) 
                SET d.swift_ref = row.swift_msg_id,
                    d.price_deviation = toFloat(row.unit_price) / toFloat(row.market_avg),
                    d.weight_gap_score = toFloat(row.weight_gap_score) / toFloat(row.declared_weight_kg),
                    d.ais_status = row.ais_status
            """, batch=chunk.to_dict('records'))

    # 3. SWIFT + Links (Remains the same)
    print(f"   💸 Loading SWIFT & Linking for {bank_name}...")
    for chunk in pd.read_csv(os.path.join(path, "swift.csv"), chunksize=CHUNK_SIZE):
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS row
                MERGE (s:Account {id: row.sender_id})
                MERGE (r:Account {id: row.receiver_id})
                CREATE (t:Transaction {id: row.msg_id})
                SET t.amount = toFloat(row.amount)
                CREATE (s)-[:SENDS]->(t)
                CREATE (t)-[:TO]->(r)
                WITH row, t
                MATCH (d:Document {swift_ref: row.msg_id})
                CREATE (t)-[:HAS_DOC]->(d)
            """, batch=chunk.to_dict('records'))
            
    driver.close()
    print(f"✅ {bank_name} finished successfully!")

if __name__ == "__main__":
    for name, cfg in BANKS.items():
        if os.path.exists(cfg['path']):
            ingest_bank(name, cfg)
        else:
            print(f"❌ ERROR: Cannot find folder '{cfg['path']}'!")