import pandas as pd
from neo4j import GraphDatabase
import os

# ==========================================
# 1. CONNECTION SETTINGS (Matches Screenshot)
# ==========================================
URI      = "bolt://127.0.0.1:7687" 
USER     = "neo4j"
PASSWORD = "password"  # <-- Change this to the password you set for 'trxn_graph'

# Folder where your bank data is stored
DATA_PATH = "./bank_a_data/" 

class TBMLGraphLoader:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
            # Test the connection
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j Instance: trxn_graph")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            exit()

    def run(self, query, **params):
        with self.driver.session() as s:
            return s.run(query, **params).data()

    def load_data(self):
        # --- A. Check if files exist ---
        files = ["kyc.csv", "swift.csv", "trade_docs_enriched.csv"]
        for f in files:
            if not os.path.exists(DATA_PATH + f):
                print(f"❌ Error: {DATA_PATH + f} not found!")
                return

        # --- B. Read Data ---
        print("📖 Reading Enriched CSVs...")
        kyc = pd.read_csv(DATA_PATH + "kyc.csv")
        swift = pd.read_csv(DATA_PATH + "swift.csv")
        trade = pd.read_csv(DATA_PATH + "trade_docs_enriched.csv")

        # --- C. Clear Old Data (Reset Graph) ---
        print("🗑  Clearing old graph nodes...")
        self.run("MATCH (n) DETACH DELETE n")

        # --- D. Load Entities (Senders/Shells) ---
        print("👤 Loading Entity Nodes...")
        self.run("""
        UNWIND $rows AS row
        MERGE (e:Entity {entity_id: row.entity_id})
        SET e.is_shell = row.is_shell,
            e.jurisdiction = row.jurisdiction,
            e.device_hash = row.device_hash
        """, rows=kyc.to_dict('records'))

        # --- E. Load Transactions (SWIFT) ---
        print("💸 Linking Transactions to Entities...")
        self.run("""
        UNWIND $rows AS row
        MATCH (e:Entity {entity_id: row.sender_id})
        CREATE (t:Transaction {msg_id: row.msg_id})
        SET t.amount = row.amount,
            t.timestamp = row.timestamp,
            t.receiver = row.receiver_id
        MERGE (e)-[:SENDS]->(t)
        """, rows=swift.to_dict('records'))

        # --- F. Load Documents (Phantom Shipping Logic) ---
        print("🚢 Connecting Trade Documents (Phantom Shipping)...")
        # Ensure ghost vessel flag is consistent
        trade['is_ghost'] = (trade['vessel_id'] == 'GHOST_VESSEL_X').astype(int)
        
        self.run("""
        UNWIND $rows AS row
        MATCH (t:Transaction {msg_id: row.swift_msg_id})
        CREATE (d:Document {doc_id: row.doc_id})
        SET d.commodity = row.commodity,
            d.weight_gap = row.weight_gap_score,
            d.price_dev = row.price_deviation,
            d.ghost = row.is_ghost
        MERGE (t)-[:HAS_DOCUMENT]->(d)
        """, rows=trade.to_dict('records'))

        # --- G. Detect Coordinated Networks (Shared Device) ---
        print("🕵️‍♂️ Finding entities sharing devices...")
        self.run("""
        MATCH (e1:Entity), (e2:Entity)
        WHERE e1.device_hash = e2.device_hash 
          AND e1.entity_id < e2.entity_id
        MERGE (e1)-[:SHARES_DEVICE_WITH]->(e2)
        """)

        print("\n🏆 SUCCESS: All Bank Data Loaded into Neo4j!")

    def close(self):
        self.driver.close()

# ==========================================
# 2. RUN THE LOADER
# ==========================================
if __name__ == "__main__":
    loader = TBMLGraphLoader()
    loader.load_data()
    loader.close()