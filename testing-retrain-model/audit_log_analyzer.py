from neo4j import GraphDatabase

# --- Configuration ---
URI = "bolt://localhost:4000"
USER = "" 
PASSWORD = ""

def calculate_crs(row):
    # w1: Sleepy Shell (25%)
    shell_score = 0
    if (row['Partner_Dormancy'] or 0) > 180: shell_score += 0.5
    if (row['Partner_Entropy'] or 0) > 0.5: shell_score += 0.5
    w1 = 0.25 * shell_score

    # w2: TBML Factors (40%)
    tbml_score = 0
    if row['Price_Dev_Ratio'] and row['Price_Dev_Ratio'] > 3.0: tbml_score += 0.4
    if row['Weight_Gap_Ratio'] and row['Weight_Gap_Ratio'] > 0.15: tbml_score += 0.3
    if row['Doc_AIS'] == 'DARK': tbml_score += 0.3
    w2 = 0.40 * tbml_score

    # w3: Topology Patterns (35%)
    w3 = 0.35 if row['Pattern_Type'] != 'LEGIT' else 0

    return round(w1 + w2 + w3, 4)

def run_audit(target_tx):
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    # Query to get primary transaction and the receiver's full history
    query = """
    MATCH (t_primary:Transaction {id: $tx_id})
    MATCH (sender:Account)-[:SENDS]->(t_primary)-[:TO]->(receiver:Account)
    
    // Find all network neighbors of that receiver
    MATCH (receiver)-[:SENDS|TO]-(other_t:Transaction)-[:SENDS|TO]-(partner:Account)
    WHERE partner <> receiver
    OPTIONAL MATCH (other_t)-[:HAS_DOC]->(d:Document)

    RETURN 
        sender.id AS Primary_Sender,
        receiver.id AS Primary_Receiver,
        other_t.id AS Tx_ID,
        other_t.pattern_type AS Pattern_Type,
        partner.id AS Partner_ID,
        partner.jurisdiction AS Partner_Loc,
        partner.is_shell AS Partner_Is_Shell,
        partner.dorm_days AS Partner_Dormancy,
        partner.device_entropy AS Partner_Entropy,
        d.price_deviation AS Price_Dev_Ratio,
        d.weight_gap_score AS Weight_Gap_Ratio,
        d.ais_status AS Doc_AIS
    """

    try:
        with driver.session() as session:
            records = session.run(query, tx_id=target_tx)
            results = [record.data() for record in records]
            if not results: return None
            
            # Enrich with Risk Scores
            for r in results:
                r['Risk_Score'] = calculate_crs(r)
            return results
    finally:
        driver.close()