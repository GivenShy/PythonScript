import json
import csv
import re
import glob
from dataclasses import dataclass
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# ------------------------------
# 🧱 Product class
# ------------------------------
@dataclass
class Product:
    name: str
    url: str
    price: float
    store: str
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


# ------------------------------
# 📥 Load products from JSON
# ------------------------------
def load_products(json_files: List[str]) -> List[Product]:
    products = []
    for file in json_files:
        store_name = file.split('.')[0]
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                volume = extract_volume(item["name"])
                products.append(Product(
                    name=item["name"],
                    url=item["url"],
                    price=item["price"],
                    store=store_name,
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


# ------------------------------
# 🔗 Group similar products
# ------------------------------
# def group_similar_products(products: List[Product], embeddings, threshold=0.80):
#     sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
#     n = len(products)
#     used = set()
#     groups = []

#     for i in range(n):
#         if i in used:
#             continue

#         current = products[i]
#         group_candidates = []

#         # Compare against others
#         for j in range(i + 1, n):
#             if sim_matrix[i][j] >= threshold:
#                 group_candidates.append((j, sim_matrix[i][j]))

#         # Always include the base product
#         group_raw = [current]
#         stores_seen = {current.store}
#         volumes_seen = {current.volume}

#         for j, score in sorted(group_candidates, key=lambda x: -x[1]):
#             candidate = products[j]

#             if candidate.store in stores_seen:
#                 continue  # only one per store

#             if current.volume and candidate.volume and candidate.volume != current.volume:
#                 continue  # only match if same volume (or volume missing)

#             group_raw.append(candidate)
#             stores_seen.add(candidate.store)
#             volumes_seen.add(candidate.volume)

#         # Mark used
#         group_indices = [products.index(p) for p in group_raw]
#         used.update(group_indices)
#         groups.append(group_raw)

#     return groups

def group_top_n_with_threshold(products: List[Product], embeddings, threshold=0.80, max_group_size=3):
    sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
    n = len(products)
    used = set()
    groups = []

    for i in range(n):
        if i in used:
            continue

        base = products[i]
        group = [base]
        stores_in_group = {base.store}
        used.add(i)

        candidates = []

        for j in range(n):
            if i == j:
                continue

            candidate = products[j]
            sim = sim_matrix[i][j]

            if sim < threshold:
                continue

            if candidate.store in stores_in_group:
                continue

            if base.volume and candidate.volume and base.volume != candidate.volume:
                continue

            candidates.append((j, sim))

        # Sort by similarity and pick top remaining slots
        candidates.sort(key=lambda x: -x[1])
        for j, sim in candidates:
            if len(group) >= max_group_size:
                break

            candidate = products[j]

            if candidate.store not in stores_in_group:
                group.append(candidate)
                used.add(j)
                stores_in_group.add(candidate.store)

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


# ------------------------------
# 🚀 Main
# ------------------------------
if __name__ == "__main__":
    json_files = glob.glob("super*.json")
    products = load_products(json_files)
    model, embeddings = generate_embeddings(products)
    groups = group_top_n_with_threshold(products, embeddings, threshold=0.80)#//group_similar_products(products, embeddings, threshold=0.80)

    for i, group in enumerate(groups, start=1):
        print(f"\nGroup {i}")
        for p in group:
            print(f"{p.store} | {p.name} | ${p.price} | {p.volume} | {p.url}")

    export_grouped_products(groups)
