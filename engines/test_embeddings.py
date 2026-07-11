import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

from testing_engine import get_embedding, get_embeddings

def test_nvidia_embeddings():
    print(f"Using NVIDIA_API_KEY: {'[SET]' if os.getenv('NVIDIA_API_KEY') else '[NOT SET]'}")
    
    print("\n--- 1. Testing single get_embedding() ---")
    try:
        embedding = get_embedding("What is the capital of France?")
        print(f"Success! Generated embedding of length: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error during get_embedding: {e}")
        
    print("\n--- 2. Testing batch get_embeddings() ---")
    try:
        texts = [
            "Paris is the capital of France.", 
            "Berlin is the capital of Germany."
        ]
        embeddings = get_embeddings(texts)
        print(f"Success! Generated {len(embeddings)} embeddings.")
        if embeddings:
            print(f"Length of first embedding: {len(embeddings[0])}")
            print(f"First 5 values of first embedding: {embeddings[0][:5]}")
    except Exception as e:
        print(f"Error during get_embeddings: {e}")

if __name__ == "__main__":
    test_nvidia_embeddings()
