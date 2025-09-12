import json
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import Optional, Dict, Any, List
import os
import time

def initialize_pinecone(
    api_key: str,
    index_name: str = "baa-embeddings",
    dimension: int = 768,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    recreate_index: bool = True
) -> tuple:
    """
    Initialize Pinecone connection and create/recreate an index.

    Args:
        api_key (str): Pinecone API key.
        index_name (str): Name of the Pinecone index.
        dimension (int): Dimension of the embeddings (default: 768 for Gemini text-embedding-004).
        metric (str): Distance metric for the index ("cosine", "euclidean", "dotproduct").
        cloud (str): Cloud provider ("aws", "gcp", "azure").
        region (str): Cloud region.
        recreate_index (bool): If True, delete existing index and create new one.

    Returns:
        tuple: (Pinecone client, Index object)
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    print(f"✅ Connected to Pinecone")
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    index_exists = index_name in existing_indexes
    
    if index_exists:
        index_stats = pc.Index(index_name).describe_index_stats()
        total_vectors = index_stats.get('total_vector_count', 0)
        print(f"📊 Found existing index '{index_name}' with {total_vectors} vectors")
    else:
        print(f"📋 Index '{index_name}' does not exist")
    
    # Handle index recreation
    if recreate_index and index_exists:
        try:
            pc.delete_index(index_name)
            print(f"🗑️ Deleted existing index '{index_name}'")
            # Wait for deletion to complete
            while index_name in pc.list_indexes().names():
                time.sleep(1)
            index_exists = False
        except Exception as e:
            print(f"⚠️ Warning: Could not delete index '{index_name}': {e}")
    
    # Create index if it doesn't exist or was just deleted
    if not index_exists or recreate_index:
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            print(f"✅ Created new index '{index_name}' with dimension {dimension}")
            
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Failed to create index '{index_name}': {e}")
            raise
    else:
        print(f"✅ Using existing index '{index_name}'")
    
    # Get index object
    index = pc.Index(index_name)
    
    return pc, index

def clear_index_data(
    index,
    namespace: str = ""
) -> bool:
    """
    Clear all data from an index namespace.
    
    Args:
        index: Pinecone index object.
        namespace (str): Namespace to clear (empty string for default namespace).
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Delete all vectors in the namespace
        index.delete(delete_all=True, namespace=namespace)
        print(f"🗑️ Cleared all vectors from namespace '{namespace or 'default'}'")
        return True
    except Exception as e:
        print(f"❌ Error clearing index data: {e}")
        return False

def get_index_stats(
    index,
    namespace: str = ""
) -> Dict[str, Any]:
    """
    Get statistics about the Pinecone index.
    
    Args:
        index: Pinecone index object.
        namespace (str): Namespace to get stats for.
    
    Returns:
        Dict[str, Any]: Index statistics or empty dict if error.
    """
    try:
        stats = index.describe_index_stats()
        
        # Extract namespace-specific stats if namespace is provided
        if namespace and 'namespaces' in stats:
            namespace_stats = stats['namespaces'].get(namespace, {})
            return {
                'total_vector_count': stats.get('total_vector_count', 0),
                'namespace_vector_count': namespace_stats.get('vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0),
                'namespaces': list(stats.get('namespaces', {}).keys())
            }
        else:
            return {
                'total_vector_count': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0),
                'namespaces': list(stats.get('namespaces', {}).keys())
            }
    except Exception as e:
        print(f"❌ Error getting index stats: {e}")
        return {}

def store_embeddings_in_pinecone(
    json_file: str, 
    index, 
    expected_dimension: int = 768,
    batch_size: int = 100,
    namespace: str = ""
) -> bool:
    """
    Store embeddings from a JSON file into a Pinecone index, with batch processing.
    Processes all items, including those with empty or zero embeddings, by assigning a default embedding.

    Args:
        json_file (str): Path to the JSON file containing code chunks and embeddings.
        index: Pinecone index object to store embeddings.
        expected_dimension (int): Expected dimension of embeddings.
        batch_size (int): Number of items to process in each batch.
        namespace (str): Namespace to store vectors in.

    Returns:
        bool: True if at least one item was stored successfully, False if file is invalid or no items could be processed.
    """
    try:
        # Validate JSON file existence
        if not os.path.exists(json_file):
            print(f"❌ File not found: {json_file}")
            return False

        # Load JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"❌ JSON must contain a list of items: {json_file}")
            return False

        if not data:
            print(f"❌ No items found in JSON file: {json_file}")
            return False

        print(f"📄 Processing {len(data)} items from {json_file}")
        
        stored_count = 0
        default_embedding_count = 0
        small_value = 1e-10  # Consistent with gemini_embedder.py
        
        # Define default embedding
        default_embedding = [0.0] * expected_dimension
        default_embedding[-1] = small_value

        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            vectors_to_upsert = []
            
            for item in batch:
                # Handle non-dictionary items
                if not isinstance(item, dict):
                    print(f"⚠️ Non-dictionary item at index {i + batch.index(item)}, assigning default embedding")
                    item = {
                        "id": f"invalid_item_{i + batch.index(item)}",
                        "embedding": default_embedding,
                        "file_name": "",
                        "file_path": "",
                        "code_chunk": "",
                        "type": "invalid",
                        "item_name": f"Invalid Item {i + batch.index(item)}",
                        "global_start_line": 0,
                        "global_end_line": 0,
                        "local_start_line": 0,
                        "local_end_line": 0,
                        "cell_number": 0
                    }
                    default_embedding_count += 1
                else:
                    # Ensure required fields
                    if "id" not in item:
                        item["id"] = f"missing_id_{i + batch.index(item)}"
                        print(f"⚠️ Missing 'id' for item {item.get('item_name', 'unknown')}, assigned default ID")
                    if "embedding" not in item:
                        item["embedding"] = default_embedding
                        print(f"⚠️ Missing 'embedding' for item {item.get('item_name', 'unknown')}, assigned default embedding")
                        default_embedding_count += 1

                # Validate embedding
                embedding = item.get("embedding")
                if not isinstance(embedding, list) or len(embedding) != expected_dimension:
                    print(f"⚠️ Invalid embedding dimension for item {item.get('item_name', 'unknown')}, assigning default embedding")
                    embedding = default_embedding
                    default_embedding_count += 1
                elif all(abs(x) < 1e-10 for x in embedding):
                    print(f"⚠️ Zero embedding for item {item.get('item_name', 'unknown')}, assigning default embedding")
                    embedding = default_embedding
                    default_embedding_count += 1

                # Prepare Pinecone vector data
                vector_data = {
                    "id": str(item["id"]),
                    "values": embedding,
                    "metadata": {
                        "file_name": str(item.get("file_name", "")),
                        "file_path": str(item.get("file_path", "")),
                        "code_chunk": str(item.get("code_chunk", ""))[:40000],  # Pinecone metadata limit
                        "type": str(item.get("type", "")),
                        "item_name": str(item.get("item_name", "")),
                        "global_start_line": int(item.get("global_start_line", 0)),
                        "global_end_line": int(item.get("global_end_line", 0)),
                        "local_start_line": int(item.get("local_start_line", 0)),
                        "local_end_line": int(item.get("local_end_line", 0)),
                        "cell_number": int(item.get("cell_number", 0))
                    }
                }
                
                vectors_to_upsert.append(vector_data)

            # Upsert batch to Pinecone
            if vectors_to_upsert:
                try:
                    index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                    stored_count += len(vectors_to_upsert)
                    print(f"📦 Processed batch {i//batch_size + 1}: {len(vectors_to_upsert)} vectors stored")
                except Exception as e:
                    print(f"❌ Error upserting batch {i//batch_size + 1}: {e}")

        # Report results
        if stored_count > 0:
            print(f"✅ Successfully stored {stored_count} embeddings from {json_file}")
            if default_embedding_count > 0:
                print(f"⚠️ Assigned default embeddings to {default_embedding_count} items")
            return True
        else:
            print(f"❌ No items could be stored from {json_file} ({default_embedding_count} default embeddings assigned)")
            return False

    except Exception as e:
        print(f"❌ Error storing embeddings for {json_file}: {e}")
        return False

def query_similar_vectors(
    index,
    query_vector: List[float],
    top_k: int = 10,
    namespace: str = "",
    include_metadata: bool = True,
    include_values: bool = False,
    filter_dict: Optional[Dict] = None
) -> List[Dict]:
    """
    Query for similar vectors in the Pinecone index.
    
    Args:
        index: Pinecone index object.
        query_vector (List[float]): Query vector.
        top_k (int): Number of similar vectors to return.
        namespace (str): Namespace to query.
        include_metadata (bool): Whether to include metadata in results.
        include_values (bool): Whether to include vector values in results.
        filter_dict (Optional[Dict]): Metadata filter conditions.
    
    Returns:
        List[Dict]: List of similar vectors with scores and metadata.
    """
    try:
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=include_metadata,
            include_values=include_values,
            filter=filter_dict
        )
        
        return response.get('matches', [])
    except Exception as e:
        print(f"❌ Error querying vectors: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Set your Pinecone API key
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Set this environment variable
    
    if not PINECONE_API_KEY:
        print("❌ Please set PINECONE_API_KEY environment variable")
        exit(1)
    
    # Initialize Pinecone connection with index recreation
    pc, index = initialize_pinecone(
        api_key=PINECONE_API_KEY,
        index_name="baa-embeddings",
        dimension=768,
        recreate_index=True  # This will delete and recreate the index
    )
    
    # Get index stats
    stats = get_index_stats(index)
    print(f"📊 Index stats: {stats}")
    
    # Store embeddings from JSON file
    success = store_embeddings_in_pinecone(
        json_file="example_py.json",
        index=index
    )
    
    # Example query (uncomment to test)
    # query_vector = [0.1] * 768  # Example query vector
    # results = query_similar_vectors(index, query_vector, top_k=5)
    # print(f"Query results: {results}")