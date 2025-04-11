import json
import csv
import re
import glob
from dataclasses import dataclass
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import psycopg2
import matplotlib.pyplot as plt 


# ------------------------------
# 🧱 Product class
# ------------------------------
@dataclass
class Product:
    name: str
    url: str
    price: float
    store: str
    subcategory_url: str
    image_url: Optional[str] = None
    embedding: Optional[any] = None
    volume: Optional[str] = None
# ------------------------------
# 🧪 Volume extractor
# ------------------------------
def extract_volume(text: str) -> Optional[str]:
    text = text.lower()
    match = re.search(r"(\d+(?:[.,]\d+)?)\s?(ml|l|g|kg)", text)
    if match:
        num, unit = match.groups()
        num = float(num.replace(',', '.'))
        if unit == "l":
            num *= 1000
            unit = "ml"
        elif unit == "kg":
            num *= 1000
            unit = "g"
        return f"{int(num)}{unit}"
    return None

def clean_price(price_str: str) -> float:
    
    # Remove non-numeric characters and convert to float
    return float(re.sub(r"[^\d.]", "", price_str.replace('\xa0', '')))

    

# ------------------------------
# 📦 Load products from JSON files

def load_products(json_files: List[str]) -> List[Product]:
    products = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                volume = extract_volume(item["name"])
                try:
                    price = clean_price(item["price"])
                except Exception as e:
                    print(item["name"])
                    raise
                products.append(Product(
                    name=item["name"],
                    url=item["url"],
                    price=price,
                    store=item["store"],
                    subcategory_url=item["scrapeCategory"]["url"],
                    image_url=item.get("imageUrl"),
                    volume=volume
                ))
    return products


# ------------------------------
# 🤖 Generate embeddings
# ------------------------------
def generate_embeddings(products: List[Product], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    names = [p.name for p in products]
    embeddings = model.encode(names, convert_to_tensor=True)
    for i, emb in enumerate(embeddings):
        products[i].embedding = emb
    return model, embeddings

def print_histogram(sim_matrix):
    sim_values = sim_matrix.flatten()
    bins = 10
    # Plot the histogram
    plt.figure(figsize=(12, 8))
    plt.hist(sim_values, bins=bins, color='blue', alpha=0.1)
    plt.title("Histogram of Similarity Matrix Values")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.1)
    plt.show()

# ------------------------------
# 🔍 Group similar products
# ------------------------------
def group_top_n_with_threshold(products: List[Product], embeddings, threshold=0.80, max_group_size=3):
    sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
    n = len(products)
    used_products = set()
    groups = []

    for i in range(n):
        base = products[i]
        if id(base) in used_products:
            continue

        group = [base]
        stores_in_group = {base.store}
        used_products.add(id(base))

        candidates = []

        for j in range(n):
            if i == j:
                continue

            candidate = products[j]
            if id(candidate) in used_products:
                continue

            sim = sim_matrix[i][j]

            if sim < threshold:
                continue

            if candidate.store in stores_in_group:
                continue

            if base.volume and candidate.volume and base.volume != candidate.volume:
                continue

            candidates.append((j, sim))

        candidates.sort(key=lambda x: -x[1])
        for j, sim in candidates:
            if len(group) >= max_group_size:
                break

            candidate = products[j]
            if candidate.store not in stores_in_group:
                group.append(candidate)
                stores_in_group.add(candidate.store)
                used_products.add(id(candidate))

        groups.append(group)

    return groups


# ------------------------------
# 📤 Export to CSV & JSON
# ------------------------------
def export_grouped_products(groups, csv_path="product_groups.csv", json_path="product_groups.json"):
    rows = []
    json_output = []

    for group_id, group in enumerate(groups, start=1):
        group_data = []
        for product in group:
            row = {
                "group_id": group_id,
                "store": product.store,
                "name": product.name,
                "price": product.price,
                "url": product.url,
                "volume": product.volume
            }
            rows.append(row)
            group_data.append(row)
        json_output.append(group_data)

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)

    print(f"\nExported {len(groups)} groups to '{csv_path}' and '{json_path}'")


def generate_sql_for_groups(groups: List[List[Product]], product_meta_table="ProductMeta", product_table="Product"):
    sql_statements = []

    store_name_to_id = {
        "sas": 1,
        "yerevan_city": 2,
        "supermarket.am": 3
    }

    for group_id, group in enumerate(groups, start=1):
        base_product = next((p for p in group if p.store == "yerevan_city"), group[0])

        subcategory_url = base_product.subcategory_url
        supermarket_name = base_product.store
        product_meta_name = base_product.name.replace("'", "''")[:100]
        product_meta_image = base_product.image_url or ''

        sql_statements.append(f"""
-- Group {group_id}
WITH meta_insert AS (
    INSERT INTO "{product_meta_table}" ("Name", "Image", "Description", "SubcategoryId")
    VALUES (
        '{product_meta_name}',
        '{product_meta_image}',
        NULL,
        NULL
    )
    RETURNING "Id" AS meta_id
)
""")

        values = []
        for product in group:
            product_name = product.name.replace("'", "''")
            product_url = product.url
            price = product.price
            sub_url = product.subcategory_url
            store = product.store
            store_id = store_name_to_id[store]

            values.append(f"""(
    '{product_name}',
    {price},
    '{product_url}',
    (SELECT "Id" FROM "Subcategory" WHERE "URL" = '{sub_url}' AND "SupermarketId" = {store_id}),
    (SELECT meta_id FROM meta_insert),
    {store_id}
)""")

        joined_values = ",\n".join(values)

        sql_statements.append(f"""
INSERT INTO "{product_table}" ("Name", "Price", "URL", "SubcategoryId", "ProductMetaId", "SupermarketId")
VALUES
{joined_values};
""")

    return "\n".join(sql_statements)


# ------------------------------
# 🚀 Main
# ------------------------------
if __name__ == "__main__":
    json_files = glob.glob("products*.json")
    products = load_products(json_files)
    print(f"Loaded {len(products)} products from {len(json_files)} files")
    model, embeddings = generate_embeddings(products)
    groups = group_top_n_with_threshold(products, embeddings, threshold=0.80)#//group_similar_products(products, embeddings, threshold=0.80)

    for i, group in enumerate(groups, start=1):
        print(f"\nGroup {i}")
        
        for p in group:
            try:
                print(f"{p.store} | {p.name} | ${p.price} | {p.volume} | {p.url}")
            except Exception:
                print(p.store+"|"+p.url+"-----------------------------------------")
                
    # export_grouped_products(groups)
    # sql_script = generate_sql_for_groups(groups)
    # with open("scriptSQL.txt", "w") as file:
    #     file.write(sql_script)
    # #print(sql_script)
    # conn = psycopg2.connect(
    # dbname="shoper",
    # user="postgres",
    # password="11111111",  # ← replace this
    # host="localhost",
    # port="5432"
    # )  # connect to your DB
    # cursor = conn.cursor()

    # cursor = conn.cursor()
    # cursor.execute(sql_script)
    # conn.commit()
    # cursor.close()
    # conn.close()

