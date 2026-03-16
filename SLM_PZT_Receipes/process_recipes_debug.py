import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

# ----- 1. Load the Semantic Model -----
print("Loading model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# ----- 2. Read Recipes from NDJSON File -----
ndjson_file = "all_recipes.ndjson"  # Ensure this file is in NDJSON format
limit = 500  # For testing: process 50 recipes
print(f"Reading up to {limit} recipes from {ndjson_file} ...")

recipes_data = []
with open(ndjson_file, "r", encoding="utf-8") as f:
    count = 0
    for line in f:
        if count >= limit:
            break
        try:
            recipe_obj = json.loads(line)
            recipes_data.append(recipe_obj)
            count += 1
            snippet = recipe_obj.get("input", "")[:60].replace("\n", " ")
            print(f"Loaded recipe {count}: {snippet}...")
        except Exception as e:
            print("Error parsing line:", e)

print(f"Total loaded recipes: {len(recipes_data)}.")

# ----- 3. Process Recipes and Compute Multi-Grained Embeddings -----
documents = []
print("Processing recipes and computing embeddings...")

for idx, recipe_obj in enumerate(recipes_data):
    full_text = recipe_obj.get("input", "").strip()
    if not full_text:
        continue

    # Split full text into parts using double newlines.
    parts = [part.strip() for part in full_text.split("\n\n") if part.strip()]
    title = parts[0] if parts else ""
    
    # Extract ingredients and instructions sections.
    ingredients_section = parts[1] if len(parts) > 1 else ""
    instructions_section = parts[2] if len(parts) > 2 else ""
    
    # Remove possible prefixes.
    if ingredients_section.lower().startswith("ingredients:"):
        ingredients_section = ingredients_section[len("ingredients:"):].strip()
    if instructions_section.lower().startswith("directions:"):
        instructions_section = instructions_section[len("directions:"):].strip()
    elif instructions_section.lower().startswith("instructions:"):
        instructions_section = instructions_section[len("instructions:"):].strip()
    
    # Split sections into individual lines.
    ingredients_lines = [line.strip() for line in ingredients_section.split("\n") if line.strip()]
    instructions_lines = [line.strip() for line in instructions_section.split("\n") if line.strip()]

    print(f"\nProcessing recipe {idx+1}: {title if title else 'No Title'}")
    print(f"  Ingredient lines: {len(ingredients_lines)}")
    print(f"  Instruction lines: {len(instructions_lines)}")
    
    # Compute embeddings.
    full_recipe_emb = model.encode(full_text, convert_to_tensor=False)
    ingredients_full_emb = model.encode(ingredients_section, convert_to_tensor=False) if ingredients_section else None
    instructions_full_emb = model.encode(instructions_section, convert_to_tensor=False) if instructions_section else None

    ingredients_line_embs = []
    for line in ingredients_lines:
        emb = model.encode(line, convert_to_tensor=False)
        ingredients_line_embs.append({"line": line, "embedding": emb.tolist() if hasattr(emb, "tolist") else emb})
    
    instructions_line_embs = []
    for line in instructions_lines:
        emb = model.encode(line, convert_to_tensor=False)
        instructions_line_embs.append({"line": line, "embedding": emb.tolist() if hasattr(emb, "tolist") else emb})
    
    doc = {
        "title": title,
        "raw_input": full_text,
        "full_recipe_embedding": full_recipe_emb.tolist() if hasattr(full_recipe_emb, "tolist") else full_recipe_emb,
        "ingredients": {
            "full": ingredients_full_emb.tolist() if ingredients_full_emb is not None and hasattr(ingredients_full_emb, "tolist") else ingredients_full_emb,
            "lines": ingredients_line_embs
        },
        "instructions": {
            "full": instructions_full_emb.tolist() if instructions_full_emb is not None and hasattr(instructions_full_emb, "tolist") else instructions_full_emb,
            "lines": instructions_line_embs
        }
    }
    documents.append(doc)
    print(f"  Finished processing recipe {idx+1}.")

print(f"\nTotal processed recipes: {len(documents)}.")

# ----- 4. Compute Similarity Matrix for Full Recipe Embeddings -----
print("\nBuilding similarity matrix for full recipe embeddings...")
full_emb_list = [doc["full_recipe_embedding"] for doc in documents]
full_emb_np = np.array(full_emb_list)
similarity_matrix = util.cos_sim(full_emb_np, full_emb_np)
print("Similarity matrix shape:", similarity_matrix.shape)
print("First 5 rows of similarity matrix:")
print(similarity_matrix[:5])

# ----- 5. Compute Transferability Score between Recipe Pairs -----
# Here we assume weighting factors alpha and beta (tunable parameters).
alpha = 1.0  # weight for instructions similarity
beta = 1.0   # weight for ingredients similarity

def compute_transferability(doc1, doc2, alpha=1.0, beta=1.0):
    # Compute subspace similarities:
    if doc1["ingredients"]["full"] is not None and doc2["ingredients"]["full"] is not None:
        ingr_sim = util.cos_sim(np.array(doc1["ingredients"]["full"]), np.array(doc2["ingredients"]["full"]))
    else:
        ingr_sim = 0
    if doc1["instructions"]["full"] is not None and doc2["instructions"]["full"] is not None:
        instr_sim = util.cos_sim(np.array(doc1["instructions"]["full"]), np.array(doc2["instructions"]["full"]))
    else:
        instr_sim = 0

    # Log the individual subspace similarities:
    print("    Ingredients similarity:", ingr_sim.item() if hasattr(ingr_sim, "item") else ingr_sim)
    print("    Instructions similarity:", instr_sim.item() if hasattr(instr_sim, "item") else instr_sim)
    
    # Compute the transferability score as the maximum weighted subspace similarity.
    T = max(alpha * instr_sim, beta * ingr_sim)
    print("    Combined transferability score:", T.item() if hasattr(T, "item") else T)
    return T.item() if hasattr(T, "item") else T

# ----- 6. Compute Jaccard Index for Ingredients and Instructions -----
def compute_jaccard_index(doc1, doc2, threshold=0.8):
    # Ingredients Jaccard Index
    ingredients1 = doc1["ingredients"]["lines"]
    ingredients2 = doc2["ingredients"]["lines"]
    intersection_count_ingr = 0
    for item1 in ingredients1:
        emb1 = np.array(item1["embedding"])
        match_found = False
        for item2 in ingredients2:
            emb2 = np.array(item2["embedding"])
            cosine_sim = util.cos_sim(emb1, emb2)
            if cosine_sim >= threshold:
                match_found = True
                # Debug message for ingredient match:
                print(f"      [Debug] Ingredient match: '{item1['line']}' <-> '{item2['line']}' with cosine similarity {cosine_sim.item():.4f}")
                break
        if match_found:
            intersection_count_ingr += 1
    union_count_ingr = len(ingredients1) + len(ingredients2) - intersection_count_ingr
    jaccard_ingr = intersection_count_ingr / union_count_ingr if union_count_ingr > 0 else 0

    # Instructions Jaccard Index
    instructions1 = doc1["instructions"]["lines"]
    instructions2 = doc2["instructions"]["lines"]
    intersection_count_instr = 0
    for item1 in instructions1:
        emb1 = np.array(item1["embedding"])
        match_found = False
        for item2 in instructions2:
            emb2 = np.array(item2["embedding"])
            cosine_sim = util.cos_sim(emb1, emb2)
            if cosine_sim >= threshold:
                match_found = True
                # Debug message for instruction match:
                print(f"      [Debug] Instruction match: '{item1['line']}' <-> '{item2['line']}' with cosine similarity {cosine_sim.item():.4f}")
                break
        if match_found:
            intersection_count_instr += 1
    union_count_instr = len(instructions1) + len(instructions2) - intersection_count_instr
    jaccard_instr = intersection_count_instr / union_count_instr if union_count_instr > 0 else 0

    # Overall Jaccard index as the average of the ingredient and instruction indices
    jaccard_overall = (jaccard_ingr + jaccard_instr) / 2
    return jaccard_ingr, jaccard_instr, jaccard_overall

# ----- 7. Compute Scores for Recipe Pairs and Store Top Results -----
print("\nComputing transferability and Jaccard indices for recipe pairs:")
jaccard_threshold = 0.8  # threshold for considering ingredient/instruction matches

pair_scores = []  # list to store scores for each pair

for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        print(f"\nPair: Recipe {i+1} and Recipe {j+1}")
        T = compute_transferability(documents[i], documents[j], alpha, beta)
        jaccard_ingr, jaccard_instr, jaccard_overall = compute_jaccard_index(documents[i], documents[j], threshold=jaccard_threshold)
        print(f"Transferability score between recipe {i+1} and {j+1}: {T:.4f}")
        print(f"Jaccard index - Ingredients: {jaccard_ingr:.4f}, Instructions: {jaccard_instr:.4f}, Overall: {jaccard_overall:.4f}")
        pair_scores.append({
            "recipe1_index": i,
            "recipe2_index": j,
            "recipe1_title": documents[i]["title"],
            "recipe2_title": documents[j]["title"],
            "transferability": T,
            "jaccard_ingr": jaccard_ingr,
            "jaccard_instr": jaccard_instr,
            "jaccard_overall": jaccard_overall
        })

# ----- 8. Display Top Pairs by Transferability and Jaccard Overall -----
# Sort pairs by transferability score (descending) and Jaccard overall (descending)
top_transferability = sorted(pair_scores, key=lambda x: x["transferability"], reverse=True)[:5]
top_jaccard = sorted(pair_scores, key=lambda x: x["jaccard_overall"], reverse=True)[:5]

print("\nTop 5 Recipe Pairs by Transferability Score:")
for pair in top_transferability:
    print(f"  Recipe {pair['recipe1_index']+1} ('{pair['recipe1_title']}') and Recipe {pair['recipe2_index']+1} ('{pair['recipe2_title']}') -> Transferability: {pair['transferability']:.4f}")

print("\nTop 5 Recipe Pairs by Jaccard Overall Index:")
for pair in top_jaccard:
    print(f"  Recipe {pair['recipe1_index']+1} ('{pair['recipe1_title']}') and Recipe {pair['recipe2_index']+1} ('{pair['recipe2_title']}') -> Jaccard Overall: {pair['jaccard_overall']:.4f}")

print("\nModel implementation complete.")
