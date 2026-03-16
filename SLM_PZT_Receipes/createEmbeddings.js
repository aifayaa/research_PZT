const fs = require('fs');
const axios = require('axios');
require('dotenv').config();
const readline = require('readline');

// --- Helper: Sleep function to respect rate limits ---
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// --- Function to query the RecipeBERT model via HF Inference API ---
// It sends the input text and returns a mean-pooled embedding vector.
async function queryModel(text, hfToken) {
  const MODEL_ENDPOINT = 'https://api-inference.huggingface.co/models/alexdseo/RecipeBERT';
  try {
    console.log('Querying model for text length:', text.length);
    const response = await axios.post(
      MODEL_ENDPOINT,
      { inputs: text },
      {
        headers: {
          'Authorization': `Bearer ${hfToken}`,
          'Content-Type': 'application/json'
        }
      }
    );
    if (response.status !== 200) {
      console.error(`API error: ${response.status}`);
      return null;
    }
    const result = response.data;
    if (result.error) {
      console.error('API returned error:', result.error);
      return null;
    }
    // The API returns a nested array: [ [token1, token2, ...] ]
    const tokenEmbeddings = result[0];
    // Mean pooling over tokens to create a single fixed-length vector.
    const embedding = tokenEmbeddings[0].map((_, colIndex) => {
      return tokenEmbeddings.reduce((sum, token) => sum + token[colIndex], 0) / tokenEmbeddings.length;
    });
    console.log('Obtained embedding of length:', embedding.length);
    return embedding;
  } catch (error) {
    console.error('Error querying model:', error.message);
    return null;
  }
}

// --- Function to process recipes using line-by-line streaming ---
// This version assumes your JSON file is NDJSON (one JSON object per line).
// For each recipe, we compute three embeddings (ingredients, instructions, full recipe)
async function processRecipesStream(filePath, hfToken, limit = 5) {
  const metadata = [];
  let count = 0;
  const fileStream = fs.createReadStream(filePath);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });
  
  // Using for-await-of ensures we process lines sequentially.
  for await (const line of rl) {
    if (count >= limit) {
      console.log("Reached the limit of recipes. Breaking out of loop.");
      break;
    }
    let recipe;
    try {
      recipe = JSON.parse(line);
    } catch (err) {
      console.error("Error parsing line:", err);
      continue;
    }
    
    // Get the full recipe text from the "input" field
    const fullTextRaw = recipe.input || "";
    // Split text into parts using double newlines as delimiter.
    const parts = fullTextRaw.split(/\n\n/).map(part => part.trim()).filter(Boolean);
    // Assume first part is the title.
    const titleText = parts[0] || "";
    // For ingredients: if the second part starts with "Ingredients:" (case-insensitive), remove the prefix.
    const ingredientsText = (parts[1] && /^Ingredients:/i.test(parts[1]))
      ? parts[1].replace(/^Ingredients:\s*/i, "")
      : "";
    // For instructions: if the third part starts with "Directions:" or "Instructions:" (case-insensitive), remove the prefix.
    const instructionsText = (parts[2] && /^(Directions|Instructions):/i.test(parts[2]))
      ? parts[2].replace(/^(Directions|Instructions):\s*/i, "")
      : "";
    
    // If file doesn't follow the expected format, use full text as fallback.
    const ingredientsFinal = ingredientsText || fullTextRaw;
    const instructionsFinal = instructionsText || fullTextRaw;
    const fullRecipeText = fullTextRaw;
    
    console.log(`\nProcessing recipe ${count + 1}: ${titleText || 'No Title'}`);
    
    console.log("Querying ingredients embedding...");
    const ingredientsEmb = await queryModel(ingredientsFinal, hfToken);
    await sleep(1500);
    
    console.log("Querying instructions embedding...");
    const instructionsEmb = await queryModel(instructionsFinal, hfToken);
    await sleep(1500);
    
    console.log("Querying full recipe embedding...");
    const fullEmb = await queryModel(fullRecipeText, hfToken);
    await sleep(1500);
    
    if (!ingredientsEmb || !instructionsEmb || !fullEmb) {
      console.log('Skipping recipe due to error in embeddings.');
      continue;
    } else {
      metadata.push({
        recipe_id: count,
        title: titleText,
        text: fullRecipeText,
        embeddings: {
          ingredients: ingredientsEmb,
          instructions: instructionsEmb,
          full_recipe: fullEmb
        }
      });
      console.log(`Finished processing recipe ${count + 1}.`);
      count++;
    }
  }
  
  rl.close();
  console.log(`Finished processing ${count} recipes.`);
  return metadata;
}

// --- Utility: Compute cosine similarity between two vectors ---
function cosineSimilarity(vec1, vec2) {
  const dot = vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
  const norm1 = Math.sqrt(vec1.reduce((sum, v) => sum + v * v, 0));
  const norm2 = Math.sqrt(vec2.reduce((sum, v) => sum + v * v, 0));
  return dot / (norm1 * norm2);
}

// --- Example: Compare similarity between two recipes for a given embedding type ---
function compareRecipes(metadata, index1, index2, embeddingType = 'full_recipe') {
  if (index1 >= metadata.length || index2 >= metadata.length) {
    console.error('Invalid indices for comparison.');
    return;
  }
  const vec1 = metadata[index1].embeddings[embeddingType];
  const vec2 = metadata[index2].embeddings[embeddingType];
  const sim = cosineSimilarity(vec1, vec2);
  console.log(`Cosine similarity between recipe ${index1} and recipe ${index2} (${embeddingType}): ${sim.toFixed(4)}`);
}

// --- Main pipeline ---
async function main() {
  // Set your Hugging Face API key here, or use a .env file.
  const HF_API_TOKEN = process.env.HF_API_TOKEN || 'YOUR_HF_API_TOKEN';
  console.log('Using HF_API_TOKEN:', HF_API_TOKEN);
  
  const datasetFile = 'all_recipes.json';  // Path to your NDJSON dataset file.
  const limit = 5;  // Only process 5 recipes for debugging.
  
  const recipesMetadata = await processRecipesStream(datasetFile, HF_API_TOKEN, limit);
  console.log(`\nProcessed ${recipesMetadata.length} recipes with three embeddings each.`);
  
  // Compare embeddings for the first two recipes (if available)
  if (recipesMetadata.length >= 2) {
    compareRecipes(recipesMetadata, 0, 1, 'ingredients');
    compareRecipes(recipesMetadata, 0, 1, 'instructions');
    compareRecipes(recipesMetadata, 0, 1, 'full_recipe');
  }
}

main().catch(err => console.error(err));
