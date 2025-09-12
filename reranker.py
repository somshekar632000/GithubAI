import os
import json
import requests
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

class NVIDIAReranker:
    """Reranker using NVIDIA's dedicated reranking API for semantic search results"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key is required. Set NVIDIA_API_KEY environment variable or pass it directly.")
        
        self.invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        
        # Create a session for connection reuse
        self.session = requests.Session()
    
    def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank code chunks using NVIDIA's dedicated reranking API
        
        Args:
            query: The original user query
            chunks: List of code chunks to rerank (should be top 25 from semantic search)
            top_k: Number of top chunks to return (default: 10)
            
        Returns:
            List of reranked chunks (top_k)
        """
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            return chunks
        
        try:
            # Prepare passages for reranking
            passages = self._prepare_passages(chunks)
            
            # Create payload for NVIDIA reranking API
            payload = {
                "model": "nvidia/nv-rerankqa-mistral-4b-v3",
                "query": {
                    "text": query
                },
                "passages": passages
            }
            
            # Make request to NVIDIA reranking API
            response = self.session.post(self.invoke_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse response and apply reranking
            response_body = response.json()
            return self._apply_reranking_from_response(chunks, response_body, top_k)
            
        except Exception as e:
            print(f"Error in reranking: {e}")
            print("Falling back to original order...")
            return chunks[:top_k]
    
    def _prepare_passages(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Prepare chunks as passages for the reranking API"""
        passages = []
        
        for chunk in chunks:
            # Create a comprehensive text representation of the chunk
            file_info = chunk.get('file_path') or chunk.get('file_name', 'Unknown')
            item_name = chunk.get('item_name', 'N/A')
            chunk_type = chunk.get('type', 'code')
            lines = f"{chunk.get('global_start_line', 0)}-{chunk.get('global_end_line', 0)}"
            code_chunk = chunk.get('code_chunk', '')
            
            # Create a rich text representation for better reranking
            passage_text = f"""File: {file_info}
Type: {chunk_type}
Item: {item_name}
Lines: {lines}
Code:
{code_chunk}"""
            
            passages.append({"text": passage_text})
        
        return passages
    
    def _apply_reranking_from_response(self, chunks: List[Dict[str, Any]], response_body: Dict, top_k: int) -> List[Dict[str, Any]]:
        """Apply reranking based on NVIDIA API response"""
        try:
            # Extract rankings from response
            rankings = response_body.get('rankings', [])
            
            if not rankings:
                print("No rankings found in response, returning original order")
                return chunks[:top_k]
            
            # Sort rankings by relevance score (descending)
            sorted_rankings = sorted(rankings, key=lambda x: x.get('relevance', 0), reverse=True)
            
            # Apply reranking
            reranked_chunks = []
            for ranking in sorted_rankings[:top_k]:
                index = ranking.get('index', 0)
                if 0 <= index < len(chunks):
                    # Add relevance score to chunk for reference
                    chunk_with_score = chunks[index].copy()
                    chunk_with_score['rerank_score'] = ranking.get('relevance', 0)
                    reranked_chunks.append(chunk_with_score)
            
            return reranked_chunks
            
        except Exception as e:
            print(f"Error applying reranking from response: {e}")
            return chunks[:top_k]
    
    def rerank_with_scores(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank chunks and return with relevance scores
        
        Returns:
            List of (chunk, relevance_score) tuples
        """
        reranked_chunks = self.rerank_chunks(query, chunks, top_k)
        
        # Extract scores from reranked chunks
        scored_chunks = []
        for chunk in reranked_chunks:
            score = chunk.get('rerank_score', 0.0)
            # Remove the score from the chunk to keep it clean
            clean_chunk = {k: v for k, v in chunk.items() if k != 'rerank_score'}
            scored_chunks.append((clean_chunk, score))
        
        return scored_chunks
    
    def get_detailed_rankings(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get detailed ranking information for all chunks
        
        Returns:
            List of chunks with detailed ranking information
        """
        if not chunks:
            return []
        
        try:
            # Prepare passages for reranking
            passages = self._prepare_passages(chunks)
            
            # Create payload for NVIDIA reranking API
            payload = {
                "model": "nvidia/nv-rerankqa-mistral-4b-v3",
                "query": {
                    "text": query
                },
                "passages": passages
            }
            
            # Make request to NVIDIA reranking API
            response = self.session.post(self.invoke_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse response and add ranking info to all chunks
            response_body = response.json()
            rankings = response_body.get('rankings', [])
            
            # Create mapping of index to ranking info
            ranking_map = {r.get('index', 0): r for r in rankings}
            
            # Add ranking info to chunks
            detailed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_with_ranking = chunk.copy()
                ranking_info = ranking_map.get(i, {})
                chunk_with_ranking['rerank_score'] = ranking_info.get('relevance', 0.0)
                chunk_with_ranking['original_index'] = i
                detailed_chunks.append(chunk_with_ranking)
            
            # Sort by relevance score
            detailed_chunks.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            return detailed_chunks
            
        except Exception as e:
            print(f"Error getting detailed rankings: {e}")
            # Return original chunks with zero scores
            return [dict(chunk, rerank_score=0.0, original_index=i) for i, chunk in enumerate(chunks)]

# Integration functions for the main chatbot
def rerank_semantic_search_results(query: str, chunks: List[Dict[str, Any]], 
                                 nvidia_api_key: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank semantic search results using NVIDIA's dedicated API
    
    Args:
        query: User query
        chunks: Code chunks from semantic search (should be top 25)
        nvidia_api_key: NVIDIA API key (optional if set in environment)
        top_k: Number of top chunks to return
        
    Returns:
        Reranked chunks (top_k)
    """
    try:
        reranker = NVIDIAReranker(nvidia_api_key)
        return reranker.rerank_chunks(query, chunks, top_k)
    except Exception as e:
        print(f"Reranking failed: {e}")
        print("Falling back to original semantic search order...")
        return chunks[:top_k]

def rerank_with_detailed_scores(query: str, chunks: List[Dict[str, Any]], 
                               nvidia_api_key: str = None, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
    """
    Convenience function to rerank semantic search results with detailed scores
    
    Args:
        query: User query
        chunks: Code chunks from semantic search
        nvidia_api_key: NVIDIA API key (optional if set in environment)
        top_k: Number of top chunks to return
        
    Returns:
        List of (chunk, relevance_score) tuples
    """
    try:
        reranker = NVIDIAReranker(nvidia_api_key)
        return reranker.rerank_with_scores(query, chunks, top_k)
    except Exception as e:
        print(f"Reranking with scores failed: {e}")
        print("Falling back to original semantic search order...")
        return [(chunk, 0.0) for chunk in chunks[:top_k]]

# Test function
def test_reranker():
    """Test the reranker with sample data"""
    # Sample chunks for testing
    sample_chunks = [
        {
            "file_path": "models/neural_network.py",
            "type": "function",
            "item_name": "forward",
            "global_start_line": 10,
            "global_end_line": 20,
            "code_chunk": "def forward(self, x):\n    return self.layer(x)"
        },
        {
            "file_path": "utils/data_loader.py", 
            "type": "function",
            "item_name": "load_data",
            "global_start_line": 5,
            "global_end_line": 15,
            "code_chunk": "def load_data(path):\n    return pd.read_csv(path)"
        },
        {
            "file_path": "models/neural_network.py",
            "type": "class",
            "item_name": "NeuralNetwork",
            "global_start_line": 1,
            "global_end_line": 50,
            "code_chunk": "class NeuralNetwork:\n    def __init__(self):\n        self.layer = nn.Linear(10, 1)"
        },
        {
            "file_path": "models/neural_network.py",
            "type": "function",
            "item_name": "backward",
            "global_start_line": 21,
            "global_end_line": 30,
            "code_chunk": "def backward(self, grad):\n    return grad * self.weight"
        }
    ]
    
    query = "How does the neural network forward pass work?"
    
    try:
        print("Testing NVIDIA reranking API...")
        
        # Test basic reranking
        reranked = rerank_semantic_search_results(query, sample_chunks, top_k=3)
        print(f"Original chunks: {len(sample_chunks)}")
        print(f"Reranked chunks: {len(reranked)}")
        
        print("\nReranked results:")
        for i, chunk in enumerate(reranked):
            score = chunk.get('rerank_score', 'N/A')
            print(f"{i+1}. {chunk['file_path']} - {chunk['item_name']} ({chunk['type']}) - Score: {score}")
        
        print("\nTesting with detailed scores...")
        # Test with detailed scores
        scored_results = rerank_with_detailed_scores(query, sample_chunks, top_k=3)
        
        print("Scored results:")
        for i, (chunk, score) in enumerate(scored_results):
            print(f"{i+1}. {chunk['file_path']} - {chunk['item_name']} ({chunk['type']}) - Score: {score:.4f}")
        
        # Test detailed rankings for all chunks
        print("\nDetailed rankings for all chunks:")
        reranker = NVIDIAReranker()
        detailed_rankings = reranker.get_detailed_rankings(query, sample_chunks)
        
        for i, chunk in enumerate(detailed_rankings):
            score = chunk.get('rerank_score', 0)
            original_idx = chunk.get('original_index', 0)
            print(f"Rank {i+1}: {chunk['file_path']} - {chunk['item_name']} (Original #{original_idx+1}) - Score: {score:.4f}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reranker()