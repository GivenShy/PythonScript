import json
import psycopg2

# CONFIGURATION
SUPERMARKET_ID = 2  # Change this to your actual supermarket ID
JSON_PATH = "./categories_en.json"

# PostgreSQL connection parameters
conn = psycopg2.connect(
    dbname="shoper",
    user="postgres",
    password="11111111",  # ← replace this
    host="localhost",
    port="5432"
)

cursor = conn.cursor()

# Read JSON data
with open(JSON_PATH, 'r', encoding='utf-8') as file:
    categories = json.load(file)

inserted_count = 0

for category in categories:
    name = category.get("name", "").strip()
    url = category.get("url", "").strip()

    if not name or not url:
        print(f"Skipping category due to missing name or url: {category}")
        continue

    cursor.execute("""
        INSERT INTO public."Subcategory" ("Name", "URL", "SupermarketId")
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;
    """, (name, url, SUPERMARKET_ID))

    inserted_count += 1

# Commit and close
conn.commit()
cursor.close()
conn.close()

print(f"Inserted {inserted_count} subcategories into the database.")
