"""Realistic Kafka transaction simulator for TBML inference.

The generator samples a latent behavior state first, then emits correlated
transaction features. This keeps the stream stochastic and avoids the hard
if/then rules that make synthetic data too easy for the model to memorize.
"""

import json
import math
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from kafka import KafkaProducer
from neo4j import GraphDatabase

KAFKA_TOPIC = "incoming_swift_messages"
KAFKA_BROKER = "localhost:9092"
NEO4J_URI = "bolt://localhost:4000"
NEO4J_USER = ""
NEO4J_PASS = ""

COMMODITY_PRICES = {
  "Textiles": 50.0,
  "Electronics": 450.0,
  "Scrap Metal": 120.0,
  "Luxury Goods": 1200.0,
  "Grain": 30.0,
}

COMMODITY_BASE_QTY = {
  "Textiles": 420,
  "Electronics": 180,
  "Scrap Metal": 260,
  "Luxury Goods": 60,
  "Grain": 900,
}

COMMODITY_UNIT_WEIGHT = {
  "Textiles": 1.2,
  "Electronics": 0.8,
  "Scrap Metal": 4.5,
  "Luxury Goods": 0.4,
  "Grain": 2.0,
}

STATE_PATTERN_WEIGHTS = {
  "routine": ["LEGIT", "LEGIT", "LEGIT", "LEGIT", "LEGIT"],
  "stressed": ["LEGIT", "SUSPICIOUS_TRANSFER", "LEGIT", "LEGIT", "CHAINED"],
  "anomalous": ["SUSPICIOUS_TRANSFER", "FAN-IN", "SPLIT_INVOICE", "CHAINED"],
  "covert": ["CYCLE", "STACK", "FAN-IN", "LAYERED", "SPLIT_INVOICE"],
}

AIS_BY_STATE = {
  "routine": ["ON", "ON", "ON", "OFF"],
  "stressed": ["ON", "ON", "OFF", "ON", "OFF"],
  "anomalous": ["ON", "OFF", "DARK", "OFF"],
  "covert": ["OFF", "DARK", "DARK", "OFF", "ON"],
}

PORT_BY_STATE = {
  "routine": ["VERIFIED", "VERIFIED", "VERIFIED", "PENDING"],
  "stressed": ["VERIFIED", "UNVERIFIED", "VERIFIED"],
  "anomalous": ["UNVERIFIED", "VERIFIED", "QUARANTINED"],
  "covert": ["UNVERIFIED", "QUARANTINED", "UNVERIFIED", "BLOCKED"],
}


@dataclass(frozen=True)
class AccountProfile:
  account_id: str
  is_shell: float
  dorm_days: float
  device_entropy: float
  jurisdiction: str


def fetch_account_profiles(driver):
  query = """
  MATCH (a:Account)
  RETURN
    coalesce(a.id, '') AS account_id,
    coalesce(a.is_shell, 0) AS is_shell,
    coalesce(a.dorm_days, 0) AS dorm_days,
    coalesce(a.device_entropy, 0) AS device_entropy,
    coalesce(a.jurisdiction, 'UNKNOWN') AS jurisdiction
  ORDER BY account_id
  """
  profiles = []
  with driver.session() as session:
    for record in session.run(query):
      account_id = record["account_id"]
      if not account_id:
        continue
      profiles.append(
        AccountProfile(
          account_id=account_id,
          is_shell=float(record["is_shell"] or 0),
          dorm_days=float(record["dorm_days"] or 0),
          device_entropy=float(record["device_entropy"] or 0),
          jurisdiction=str(record["jurisdiction"] or "UNKNOWN"),
        )
      )
  return profiles


def fallback_profiles(count=120):
  jurisdictions = ["SG", "UK", "AE", "US", "HK"]
  profiles = []
  for index in range(count):
    profiles.append(
      AccountProfile(
        account_id=f"ACC_{index:06d}",
        is_shell=1.0 if random.random() < 0.08 else 0.0,
        dorm_days=max(0.0, random.gauss(60.0, 40.0)),
        device_entropy=min(0.99, max(0.01, random.random() ** 2)),
        jurisdiction=random.choice(jurisdictions),
      )
    )
  return profiles


def profile_risk(profile):
  shell = 0.35 if profile.is_shell else 0.0
  dorm = min(profile.dorm_days / 365.0, 1.5) * 0.28
  entropy = min(profile.device_entropy, 1.0) * 0.22
  return min(0.95, shell + dorm + entropy)


def sample_sender(profiles):
  weights = []
  for profile in profiles:
    activity = 1.0 + max(0.0, 1.0 - min(profile.dorm_days / 365.0, 1.0))
    risk = 1.0 + profile_risk(profile)
    weights.append(activity * risk)
  return random.choices(profiles, weights=weights, k=1)[0]


def sample_receiver(profiles, sender, state):
  eligible = [profile for profile in profiles if profile.account_id != sender.account_id]
  if not eligible:
    return sender

  same_jurisdiction = [profile for profile in eligible if profile.jurisdiction == sender.jurisdiction]
  cross_jurisdiction = [profile for profile in eligible if profile.jurisdiction != sender.jurisdiction]

  if state == "routine" and same_jurisdiction and random.random() < 0.7:
    pool = same_jurisdiction
  elif state == "covert" and cross_jurisdiction:
    pool = cross_jurisdiction
  else:
    pool = eligible

  weights = []
  for profile in pool:
    activity = 1.0 + max(0.0, 1.0 - min(profile.dorm_days / 365.0, 1.0))
    weights.append(activity)
  return random.choices(pool, weights=weights, k=1)[0]


def sample_behavior_state(sender, receiver):
  sender_risk = profile_risk(sender)
  receiver_risk = profile_risk(receiver)
  jurisdiction_gap = 1.0 if sender.jurisdiction != receiver.jurisdiction else 0.0
  latent_risk = min(0.95, 0.5 * sender_risk + 0.3 * receiver_risk + 0.2 * jurisdiction_gap)

  weights = [
    max(0.05, 0.68 - 0.35 * latent_risk),
    0.14 + 0.20 * latent_risk,
    0.10 + 0.22 * latent_risk,
    0.08 + 0.33 * latent_risk,
  ]
  total = sum(weights)
  probabilities = [weight / total for weight in weights]
  state = random.choices(["routine", "stressed", "anomalous", "covert"], weights=probabilities, k=1)[0]
  return state, latent_risk


def sample_commodity(state):
  if state == "routine":
    commodities = ["Textiles", "Grain", "Electronics", "Scrap Metal", "Luxury Goods"]
    weights = [4, 5, 3, 2, 1]
  elif state == "stressed":
    commodities = ["Textiles", "Electronics", "Scrap Metal", "Luxury Goods", "Grain"]
    weights = [3, 3, 3, 2, 1]
  elif state == "anomalous":
    commodities = ["Electronics", "Scrap Metal", "Luxury Goods", "Textiles", "Grain"]
    weights = [4, 4, 4, 2, 1]
  else:
    commodities = ["Luxury Goods", "Electronics", "Scrap Metal", "Textiles", "Grain"]
    weights = [5, 4, 4, 1, 1]
  return random.choices(commodities, weights=weights, k=1)[0]


def sample_trade_fields(commodity, state):
  market_avg = COMMODITY_PRICES[commodity]
  base_qty = COMMODITY_BASE_QTY[commodity]
  unit_weight = COMMODITY_UNIT_WEIGHT[commodity]

  qty_multiplier = {
    "routine": (1.0, 0.18),
    "stressed": (1.12, 0.22),
    "anomalous": (1.35, 0.30),
    "covert": (1.75, 0.38),
  }[state]
  price_multiplier = {
    "routine": (0.00, 0.04),
    "stressed": (0.10, 0.08),
    "anomalous": (0.28, 0.14),
    "covert": (0.62, 0.20),
  }[state]
  weight_gap_range = {
    "routine": (0.98, 1.00),
    "stressed": (0.92, 0.98),
    "anomalous": (0.82, 0.95),
    "covert": (0.68, 0.88),
  }[state]

  qty = int(max(1, random.lognormvariate(math.log(base_qty), 0.35) * qty_multiplier[0]))
  unit_price = market_avg * random.lognormvariate(price_multiplier[0], price_multiplier[1])
  declared_weight_kg = qty * unit_weight * random.uniform(0.97, 1.08)
  actual_weight_kg = declared_weight_kg * random.uniform(weight_gap_range[0], weight_gap_range[1])
  amount = round(qty * unit_price, 2)

  price_deviation = round(unit_price - market_avg, 2)
  weight_gap_score = round(declared_weight_kg - actual_weight_kg, 2)
  return amount, qty, unit_price, market_avg, declared_weight_kg, actual_weight_kg, price_deviation, weight_gap_score


def sample_doc_signals(state):
  ais_status = random.choices(AIS_BY_STATE[state], k=1)[0]
  port_log_status = random.choices(PORT_BY_STATE[state], k=1)[0]
  pattern_type = random.choices(STATE_PATTERN_WEIGHTS[state], k=1)[0]
  return ais_status, port_log_status, pattern_type


def build_payload(sender, receiver, state, latent_risk):
  commodity = sample_commodity(state)
  amount, qty, unit_price, market_avg, declared_weight_kg, actual_weight_kg, price_deviation, weight_gap_score = sample_trade_fields(
    commodity,
    state,
  )
  ais_status, port_log_status, pattern_type = sample_doc_signals(state)

  msg_id = f"SW_LIVE_{uuid.uuid4().hex[:8].upper()}"
  doc_id = f"DOC_LIVE_{uuid.uuid4().hex[:8].upper()}"

  return {
    "behavior_state": state,
    "latent_risk_score": round(latent_risk, 4),
    "swift": {
      "msg_id": msg_id,
      "sender_id": sender.account_id,
      "receiver_id": receiver.account_id,
      "amount": amount,
      "currency": "USD",
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "pattern_type": pattern_type,
    },
    "trade": {
      "doc_id": doc_id,
      "commodity": commodity,
      "qty": qty,
      "unit_price": round(unit_price, 2),
      "market_avg": market_avg,
      "price_deviation": price_deviation,
      "declared_weight_kg": round(declared_weight_kg, 2),
      "actual_weight_kg": round(actual_weight_kg, 2),
      "weight_gap_score": weight_gap_score,
      "ais_status": ais_status,
      "port_log_status": port_log_status,
    },
  }


def main():
  driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
  profiles = fetch_account_profiles(driver)
  if not profiles:
    profiles = fallback_profiles()

  producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    acks="all",
    linger_ms=20,
    retries=3,
  )

  print("[trigger] realistic synthetic stream started")
  print(f"[trigger] accounts loaded: {len(profiles)}")
  print("[trigger] topic: incoming_swift_messages")

  try:
    while True:
      sender = sample_sender(profiles)
      receiver = sample_receiver(profiles, sender, "routine")
      state, latent_risk = sample_behavior_state(sender, receiver)
      payload = build_payload(sender, receiver, state, latent_risk)

      producer.send(KAFKA_TOPIC, value=payload)

      swift = payload["swift"]
      trade = payload["trade"]
      print(
        f"[{time.strftime('%H:%M:%S')}] {state.upper():10s} | "
        f"{swift['msg_id']} | ${swift['amount']:,.2f} | {trade['commodity']} | "
        f"AIS:{trade['ais_status']} | Dev:{trade['price_deviation']:.2f} | Gap:{trade['weight_gap_score']:.2f}"
      )

      time.sleep(random.uniform(0.8, 2.4))

  except KeyboardInterrupt:
    print("\n[trigger] stopping synthetic stream")
  finally:
    producer.flush()
    producer.close()
    driver.close()


if __name__ == "__main__":
  main()