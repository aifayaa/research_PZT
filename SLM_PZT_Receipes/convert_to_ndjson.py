import json

input_filename = 'all_recipes.json'
output_filename = 'all_recipes.ndjson'

# Read the JSON array from the input file.
with open(input_filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Write each object as a single line to the output file.
with open(output_filename, 'w', encoding='utf-8') as f_out:
    for obj in data:
        line = json.dumps(obj)
        f_out.write(line + '\n')

print(f"Converted {len(data)} recipes from {input_filename} to NDJSON file {output_filename}.")
