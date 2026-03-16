from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("corbt/all-recipes")

# Assuming the dataset has a single split named "train"
# Save the split as a JSON file
dataset["train"].to_json("all_recipes.json")

