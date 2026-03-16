from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the lightweight model (about 33M parameters)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example recipes (you can replace these with your actual recipe texts)
recipes = [
    """No-Bake Nut Cookies
    Ingredients: 1 c. firmly packed brown sugar, 1/2 c. evaporated milk, 1/2 tsp. vanilla, 1/2 c. broken nuts, 2 Tbsp. butter, 3 1/2 c. bite-size shredded rice biscuits.
    Directions: Mix ingredients; cook on a griddle until lightly browned.""",
    
    """Jewell Ball's Chicken
    Ingredients: 1 small jar chipped beef, 4 boned chicken breasts, 1 can cream of mushroom soup, 1 carton sour cream.
    Directions: Layer beef and chicken in a baking dish; pour a mixture of soup and sour cream; bake at 275°F for 3 hours."""
]

# Generate embeddings for each recipe (runs on CPU by default)
embeddings = model.encode(recipes, convert_to_tensor=True)
print("Generated embeddings with shape:", embeddings.shape)

# Compute cosine similarity between the first two recipes
cosine_sim = util.cos_sim(embeddings[0], embeddings[1])
print("Cosine similarity between recipe 1 and recipe 2:", cosine_sim.item())

# Optionally: Using FAISS for nearest neighbor search (if you have many recipes)
# Uncomment the following lines if you want to experiment with FAISS

# import faiss
# # Convert embeddings to numpy array
# embeddings_np = embeddings.cpu().detach().numpy()
# d = embeddings_np.shape[1]
# index = faiss.IndexFlatL2(d)
# index.add(embeddings_np)
# print("FAISS index built with", index.ntotal, "vectors.")
# # Query: find top 2 nearest neighbors for the first recipe
# D, I = index.search(np.expand_dims(embeddings_np[0], axis=0), 2)
# print("Nearest neighbors (indices):", I)
# print("Distances:", D)
