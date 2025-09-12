import os
import json
import time
import shutil
import requests
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def embed_code_chunks_with_jina(
    json_file_path: str,
    api_key: Optional[str] = None,
    model: str = "jina-embeddings-v2-base-code",  # Best for coding tasks
    task: str = "retrieval.passage",  # Optimal for indexing code chunks
    batch_size: int = 100,  # Jina AI supports batching
    delay_between_requests: float = 0.2,  # Conservative delay for free tier
    expected_dimension: int = 768,  # jina-embeddings-v2-base-code dimension
    small_value: float = 1e-10
) -> bool:
    """Embed code chunks from a JSON file using Jina AI and update the file with embeddings."""
    try:
        if api_key is None:
            api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            print("❌ Error: No Jina AI API key provided. Get one free at https://jina.ai/embeddings/")
            return False

        # Test API connection
        try:
            test_response = _get_jina_embeddings(
                texts=["test"],
                api_key=api_key,
                model=model,  # Pass model parameter
                task=task
            )
            if test_response and len(test_response) == 1:
                test_embedding = test_response[0]
                if len(test_embedding) == expected_dimension:
                    print("✅ Jina AI API connection successful")
                else:
                    print(f"❌ Test embedding dimension mismatch: expected {expected_dimension}, got {len(test_embedding)}")
                    return False
            else:
                print("❌ Test API call failed")
                return False
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False

        if not os.path.exists(json_file_path):
            print(f"❌ File not found: {json_file_path}")
            return False

        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if not isinstance(data, list):
            print("❌ JSON should contain a list of items")
            return False

        if not data:
            print("❌ No items found in JSON file")
            return False

        print(f"📄 Processing {len(data)} items from: {json_file_path}")
        print(f"📏 Using {expected_dimension} dimensions for {model}")
        successful_embeddings = 0
        failed_embeddings = 0
        modified_zero_embeddings = 0
        items_to_remove = []

        # Prepare items that need embedding
        items_needing_embedding = []
        content_for_embedding = []
        
        for item_index, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"⚠️ Skipping non-dictionary item at index {item_index}")
                items_to_remove.append(item_index)
                continue

            code_chunk = item.get('code_chunk') or item.get('code') or item.get('content') or item.get('text', '')
            item_name = item.get('item_name') or item.get('name') or f'Item {item_index + 1}'

            if not code_chunk or not str(code_chunk).strip():
                print(f"⚠️ Skipping empty or invalid content: {item_name}")
                items_to_remove.append(item_index)
                modified_zero_embeddings += 1
                continue

            try:
                code_chunk = code_chunk.encode('utf-8', errors='ignore').decode('utf-8')
            except Exception as e:
                print(f"⚠️ Skipping {item_name} due to encoding error: {e}")
                items_to_remove.append(item_index)
                modified_zero_embeddings += 1
                continue

            # Check if valid embedding already exists
            if 'embedding' in item and isinstance(item['embedding'], list) and len(item['embedding']) == expected_dimension:
                if not all(x == 0.0 for x in item['embedding'][:expected_dimension-1]) or item['embedding'][-1] != small_value:
                    print(f"ℹ️ Valid embedding already exists for: {item_name}")
                    successful_embeddings += 1
                    continue
                else:
                    print(f"⚠️ Existing zero embedding found for: {item_name}")
                    items_to_remove.append(item_index)
                    modified_zero_embeddings += 1
                    continue

            # Add to batch processing list
            items_needing_embedding.append((item_index, item, item_name))
            content_for_embedding.append(code_chunk)

        # Process embeddings in batches
        if content_for_embedding:
            print(f"🚀 Processing {len(content_for_embedding)} embeddings in batches of {batch_size}...")
            
            # Process in batches
            for batch_start in range(0, len(content_for_embedding), batch_size):
                batch_end = min(batch_start + batch_size, len(content_for_embedding))
                batch_content = content_for_embedding[batch_start:batch_end]
                batch_items = items_needing_embedding[batch_start:batch_end]
                
                print(f"📡 Processing batch {batch_start//batch_size + 1}/{(len(content_for_embedding) + batch_size - 1)//batch_size} ({len(batch_content)} items)...")
                
                # Get embeddings for this batch with retries
                batch_embeddings = []
                for attempt in range(3):
                    try:
                        batch_embeddings = _get_jina_embeddings(
                            texts=batch_content,
                            api_key=api_key,
                            model=model,  # Pass model parameter
                            task=task
                        )
                        
                        if batch_embeddings and len(batch_embeddings) == len(batch_content):
                            print(f"✅ Batch completed: {len(batch_embeddings)} embeddings received")
                            break
                        else:
                            raise Exception(f"Expected {len(batch_content)} embeddings, got {len(batch_embeddings) if batch_embeddings else 0}")
                        
                    except Exception as e:
                        if attempt < 2:
                            print(f"⚠️ Batch retry {attempt + 1}/3: {e}")
                            time.sleep(delay_between_requests * (2 ** attempt))
                            continue
                        else:
                            print(f"❌ Failed to get embeddings after 3 attempts: {e}")
                            # Mark all items in this batch as failed
                            for item_index, item, item_name in batch_items:
                                items_to_remove.append(item_index)
                                failed_embeddings += 1
                            batch_embeddings = []
                            break
                
                # Process batch results
                if batch_embeddings:
                    for i, (item_index, item, item_name) in enumerate(batch_items):
                        if i < len(batch_embeddings):
                            embedding = batch_embeddings[i]
                            
                            if not isinstance(embedding, list) or len(embedding) != expected_dimension:
                                print(f"❌ Invalid embedding dimension for {item_name}: got {len(embedding) if isinstance(embedding, list) else 'non-list'}")
                                items_to_remove.append(item_index)
                                modified_zero_embeddings += 1
                                continue
                                
                            if all(abs(x) < 1e-10 for x in embedding):
                                print(f"⚠️ Zero embedding generated for: {item_name}")
                                items_to_remove.append(item_index)
                                modified_zero_embeddings += 1
                                continue

                            item['embedding'] = embedding
                            successful_embeddings += 1
                            print(f"✅ {item_name} - {len(embedding)} dimensions")
                        else:
                            print(f"❌ No embedding received for: {item_name}")
                            items_to_remove.append(item_index)
                            failed_embeddings += 1
                
                # Add delay between batches
                if batch_end < len(content_for_embedding):
                    time.sleep(delay_between_requests)
            
            print(f"🏆 All batches processed!")

        # Remove failed items
        if items_to_remove:
            print(f"🗑️ Removing {len(items_to_remove)} items with zero or invalid embeddings")
            # Sort in reverse order to avoid index shifting issues
            items_to_remove.sort(reverse=True)
            for idx in items_to_remove:
                if 0 <= idx < len(data):
                    data.pop(idx)

        # Create backup and save
        backup_path = json_file_path + '.backup'
        if os.path.exists(json_file_path):
            shutil.copy2(json_file_path, backup_path)
            print(f"📋 Backup created: {backup_path}")

        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

        print(f"\n🏁 Completed: {successful_embeddings} successful, "
              f"{modified_zero_embeddings} removed, {failed_embeddings} failed")
        return failed_embeddings == 0
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def embed_query_with_jina(
    text: str,
    api_key: Optional[str] = None,
    model: str = "jina-embeddings-v2-base-code",  # Best for coding tasks
    task: str = "retrieval.query",  # Optimal for search queries
    expected_dim: Optional[int] = None  # Will be determined dynamically
) -> Optional[List[float]]:
    """
    Generate an embedding using Jina AI's embedding API.
    
    Args:
        text: The text to embed
        api_key: Jina AI API key (if None, will use JINA_API_KEY env var)
        model: The embedding model to use (jina-embeddings-v2-base-code, etc.)
        task: The task type (text-matching, retrieval.query, retrieval.passage, etc.)
        expected_dim: Expected embedding dimension for validation (optional)
        
    Returns:
        List of floats representing the embedding, or None if failed
    """
    try:
        # Get API key
        if api_key is None:
            api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            print("❌ Error: No Jina AI API key provided. Get one free at https://jina.ai/embeddings/")
            return None

        # Generate embedding
        embeddings = _get_jina_embeddings(
            texts=[text],
            api_key=api_key,
            model=model,  # Pass model parameter
            task=task
        )
        
        if not embeddings or len(embeddings) == 0:
            print("❌ No embedding data found in response")
            return None
            
        embedding = embeddings[0]
        
        # Ensure it's a list
        if not isinstance(embedding, list):
            try:
                embedding = list(embedding)
            except Exception as e:
                print(f"❌ Error converting embedding to list: {e}")
                return None
        
        # Validate dimensions if expected_dim is provided
        if expected_dim and len(embedding) != expected_dim:
            print(f"⚠️ Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}")
        else:
            print(f"✅ Generated embedding with {len(embedding)} dimensions")
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error embedding query: {str(e)}")
        return None


def _get_jina_embeddings(
    texts: List[str],
    api_key: str,
    model: str = "jina-embeddings-v2-base-code",  # Best for coding tasks
    task: str = "retrieval.passage"  # Default for indexing
) -> Optional[List[List[float]]]:
    """
    Internal function to call Jina AI embeddings API.
    
    Args:
        texts: List of texts to embed
        api_key: Jina AI API key
        model: Model name
        task: Task type
        
    Returns:
        List of embeddings or None if failed
    """
    
    url = "https://api.jina.ai/v1/embeddings"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # For Jina AI API, input should be a list of strings, not objects
    input_texts = []
    for text in texts:
        if isinstance(text, str):
            input_texts.append(text)
        else:
            input_texts.append(str(text))
    
    # Correct payload format for Jina AI API
    payload = {
        "model": model,
        "task": task,
        "input": input_texts
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"API request failed with status {response.status_code}"
            try:
                error_detail = response.json()
                if "detail" in error_detail:
                    error_msg += f": {error_detail['detail']}"
                elif "error" in error_detail:
                    error_msg += f": {error_detail['error']}"
            except:
                error_msg += f": {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        
        if "data" not in result:
            raise Exception("No 'data' field in API response")
        
        embeddings = []
        for item in result["data"]:
            if "embedding" in item:
                embeddings.append(item["embedding"])
            else:
                raise Exception("Missing 'embedding' field in response data")
        
        return embeddings
        
    except requests.exceptions.Timeout:
        raise Exception("Request timeout - try reducing batch size")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Make sure to set your JINA_API_KEY environment variable
    # You can get a free API key at https://jina.ai/embeddings/
    
    # Example: Embed a single query (dimensions will be determined automatically)
    query_embedding = embed_query_with_jina("def hello_world(): print('Hello, World!')")
    if query_embedding:
        print(f"Query embedding generated: {len(query_embedding)} dimensions")
    
    # Example: Embed code chunks from a JSON file (dimensions determined automatically)
    # success = embed_code_chunks_with_jina("your_code_chunks.json")
    # print(f"Batch embedding {'successful' if success else 'failed'}")
