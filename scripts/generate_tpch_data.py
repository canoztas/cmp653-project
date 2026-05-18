"""Generate synthetic TPC-H-like data directly into PostgreSQL.

This avoids needing the official dbgen tool. Generates data at a given
scale factor using randomized but structurally correct data.
"""

import argparse
import random
import string
from datetime import date, timedelta
from decimal import Decimal

import psycopg2


def random_string(length):
    return "".join(random.choices(string.ascii_uppercase + " ", k=length)).strip()


def random_date(start_year=1992, end_year=1998):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
NATIONS = [
    ("ALGERIA", 0), ("ARGENTINA", 1), ("BRAZIL", 1), ("CANADA", 1),
    ("EGYPT", 0), ("ETHIOPIA", 0), ("FRANCE", 3), ("GERMANY", 3),
    ("INDIA", 2), ("INDONESIA", 2), ("IRAN", 4), ("IRAQ", 4),
    ("JAPAN", 2), ("JORDAN", 4), ("KENYA", 0), ("MOROCCO", 0),
    ("MOZAMBIQUE", 0), ("PERU", 1), ("CHINA", 2), ("ROMANIA", 3),
    ("SAUDI ARABIA", 4), ("VIETNAM", 2), ("RUSSIA", 3),
    ("UNITED KINGDOM", 3), ("UNITED STATES", 1),
]
SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]
PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
SHIPMODES = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"]
INSTRUCTIONS = ["DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"]
BRANDS = [f"Brand#{i}{j}" for i in range(1, 6) for j in range(1, 6)]
TYPES = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
CONTAINERS = ["SM CASE", "SM BOX", "SM PACK", "SM PKG", "MED BAG",
              "MED BOX", "MED PKG", "LG CASE", "LG BOX", "LG PACK",
              "WRAP CASE", "WRAP BOX"]


def generate(conn, scale_factor=0.1):
    cur = conn.cursor()

    n_suppliers = max(10, int(10000 * scale_factor))
    n_parts = max(20, int(200000 * scale_factor))
    n_customers = max(15, int(150000 * scale_factor))
    n_orders = max(60, int(1500000 * scale_factor))

    print(f"Generating TPC-H data at SF={scale_factor}...")
    print(f"  Suppliers: {n_suppliers}, Parts: {n_parts}, "
          f"Customers: {n_customers}, Orders: {n_orders}")

    # Regions
    for i, name in enumerate(REGIONS):
        cur.execute("INSERT INTO region VALUES (%s, %s, %s)",
                    (i, name, random_string(40)))

    # Nations
    for i, (name, region_key) in enumerate(NATIONS):
        cur.execute("INSERT INTO nation VALUES (%s, %s, %s, %s)",
                    (i, name, region_key, random_string(40)))

    # Suppliers
    for i in range(1, n_suppliers + 1):
        cur.execute(
            "INSERT INTO supplier VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (i, f"Supplier#{i:09d}", random_string(25),
             random.randint(0, 24), f"{random.randint(10,34)}-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
             round(random.uniform(-999.99, 9999.99), 2), random_string(60)),
        )

    # Parts
    for i in range(1, n_parts + 1):
        cur.execute(
            "INSERT INTO part VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (i, random_string(30), f"Manufacturer#{random.randint(1,5)}",
             random.choice(BRANDS), random.choice(TYPES),
             random.randint(1, 50), random.choice(CONTAINERS),
             round(random.uniform(1.0, 2000.0), 2), random_string(20)),
        )

    # PartSupp
    for part_id in range(1, n_parts + 1):
        suppliers = random.sample(range(1, n_suppliers + 1), min(4, n_suppliers))
        for supp_id in suppliers:
            cur.execute(
                "INSERT INTO partsupp VALUES (%s,%s,%s,%s,%s)",
                (part_id, supp_id, random.randint(1, 9999),
                 round(random.uniform(1.0, 1000.0), 2), random_string(80)),
            )

    # Customers
    for i in range(1, n_customers + 1):
        cur.execute(
            "INSERT INTO customer VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (i, f"Customer#{i:09d}", random_string(25),
             random.randint(0, 24),
             f"{random.randint(10,34)}-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
             round(random.uniform(-999.99, 9999.99), 2),
             random.choice(SEGMENTS), random_string(60)),
        )

    # Orders + LineItems
    lineitem_count = 0
    for i in range(1, n_orders + 1):
        cust_id = random.randint(1, n_customers)
        order_date = random_date()
        total = 0.0
        n_lines = random.randint(1, 7)

        items = []
        for j in range(1, n_lines + 1):
            qty = round(random.uniform(1, 50), 2)
            price = round(random.uniform(1, 105000 / 50), 2)
            ext_price = round(qty * price, 2)
            discount = round(random.uniform(0, 0.10), 2)
            tax = round(random.uniform(0, 0.08), 2)
            total += float(ext_price) * (1 - float(discount)) * (1 + float(tax))

            ship_date = order_date + timedelta(days=random.randint(1, 121))
            commit_date = order_date + timedelta(days=random.randint(30, 90))
            receipt_date = ship_date + timedelta(days=random.randint(1, 30))

            items.append((
                i, random.randint(1, n_parts), random.randint(1, n_suppliers), j,
                qty, ext_price, discount, tax,
                random.choice(["A", "N", "R"]),
                random.choice(["O", "F"]),
                ship_date, commit_date, receipt_date,
                random.choice(INSTRUCTIONS), random.choice(SHIPMODES),
                random_string(30),
            ))

        status = random.choice(["O", "F", "P"])
        cur.execute(
            "INSERT INTO orders VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (i, cust_id, status, round(total, 2), order_date,
             random.choice(PRIORITIES), f"Clerk#{random.randint(1, 1000):09d}",
             0, random_string(40)),
        )

        for item in items:
            cur.execute(
                "INSERT INTO lineitem VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                item,
            )
            lineitem_count += 1

        if i % 10000 == 0:
            conn.commit()
            print(f"  Orders: {i}/{n_orders}")

    conn.commit()
    print(f"  Total lineitems: {lineitem_count}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TPC-H data")
    parser.add_argument("--sf", type=float, default=0.1, help="Scale factor")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="tpch")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=args.host, port=args.port, dbname=args.dbname,
        user=args.user, password=args.password,
    )
    conn.autocommit = False
    generate(conn, args.sf)
    conn.close()
