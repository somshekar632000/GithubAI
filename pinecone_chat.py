import os
import re
import json
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, TypedDict, Annotated
import google.generativeai as genai
from dataclasses import dataclass
from gemini_embedder import embed_query_with_jina
import time
from dotenv import load_dotenv
from reranker import rerank_semantic_search_results

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# State definition for LangGraph
class ChatbotState(TypedDict):
    """State for the repository chatbot workflow"""
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    intent_analysis: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_info: Dict[str, Any]
    query_type: str
    namespace: str
    error: Optional[str]
    final_response: Optional[str]

@dataclass
class QueryResult:
    """Container for query results"""
    chunks: List[Dict[str, Any]]
    query_type: str
    retrieval_info: Dict[str, Any]

class RepoQueryProcessor:
    """Process and categorize different types of queries"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.jina_api_key = os.getenv("JINA_API_KEY")
        self.model = genai.GenerativeModel('gemini-2.0-flash-001') 
        
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and extract parameters"""
        
        prompt = f"""
        Analyze this code repository query and extract information:

        Query: "{query}"

        Look for:
        1. File paths (like folder/subfolder/file.py or just file.py)
        2. Function names (like "function_name" or "def function_name")
        3. Class names (like "class ClassName" or "ClassName class")
        4. Line numbers (like "line 10" or "lines 5-15")
        5. Cell numbers for notebooks (like "cell 3")

        Return ONLY a JSON object:
        {{
            "query_type": "line_specific|line_range|function_specific|class_specific|file_specific|cell_specific|general_semantic",
            "file_path": "full_file_path_or_null",
            "item_name": "function_or_class_name_or_null",
            "line_start": number_or_null,
            "line_end": number_or_null,
            "cell_number": number_or_null,
            "intent": "brief_description"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            
            # Post-process to ensure consistency
            result = self._post_process_analysis(result, query)
            return result
            
        except Exception as e:
            print(f"Error analyzing query intent: {e}")
            return self._fallback_intent_analysis(query)
    
    def _post_process_analysis(self, analysis: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Post-process the analysis to normalize file paths"""
        file_path = analysis.get('file_path')
        
        if file_path:
            # Normalize path separators
            file_path = file_path.replace('\\', '/')
            analysis['file_path'] = file_path
        
        return analysis
    
    def _fallback_intent_analysis(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback intent analysis using regex patterns"""
        query_lower = query.lower()
        
        # Enhanced file path patterns - normalize separators
        file_patterns = [
            r'([a-zA-Z0-9_\-/\\\.]+/[a-zA-Z0-9_\-\.]+\.(py|ipynb))',  # Full path
            r'([a-zA-Z0-9_\-\\\.]+\\[a-zA-Z0-9_\-\.]+\.(py|ipynb))',  # Windows path
            r'([a-zA-Z0-9_\-\.]+\.(py|ipynb))'  # Simple filename
        ]
        
        extracted_file_path = None
        
        for pattern in file_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted_file_path = match.group(1).replace('\\', '/')
                # Only keep as file_path if it contains directory structure
                if '/' not in extracted_file_path:
                    extracted_file_path = None
                break
        
        # Line-specific patterns
        line_patterns = [
            r'line\s+(\d+)(?:\s*-\s*(\d+))?\s+(?:in|of|from)\s+([a-zA-Z0-9_\-/\\\.]*\.(py|ipynb))',
            r'lines?\s+(\d+)(?:\s*-\s*(\d+))?\s+(?:in|of|from)\s+([a-zA-Z0-9_\-/\\\.]*\.(py|ipynb))',
            r'([a-zA-Z0-9_\-/\\\.]*\.(py|ipynb))\s+line\s+(\d+)(?:\s*-\s*(\d+))?',
            r'line\s+(\d+)(?:\s*-\s*(\d+))?',  # Line without file
        ]
        
        for pattern in line_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                groups = match.groups()
                if 'in|of|from' in pattern:
                    start_line = int(groups[0])
                    end_line = int(groups[1]) if groups[1] else start_line
                    file_ref = groups[2].replace('\\', '/') if len(groups) > 2 and groups[2] else extracted_file_path
                elif len(groups) >= 4:
                    file_ref = groups[0].replace('\\', '/')
                    start_line = int(groups[2])
                    end_line = int(groups[3]) if groups[3] else start_line
                else:
                    start_line = int(groups[0])
                    end_line = int(groups[1]) if groups[1] else start_line
                    file_ref = extracted_file_path
                
                return {
                    "query_type": "line_range" if end_line > start_line else "line_specific",
                    "file_path": file_ref if file_ref and '/' in file_ref else None,
                    "item_name": None,
                    "line_start": start_line,
                    "line_end": end_line,
                    "cell_number": None,
                    "intent": f"Explain lines {start_line}-{end_line}" + (f" in {file_ref}" if file_ref else "")
                }
        
        # Cell-specific patterns
        cell_patterns = [
            r'cell\s+(\d+)\s+(?:in|of|from)\s+([^\s]+\.ipynb)',
            r'([^\s]+\.ipynb)\s+cell\s+(\d+)',
            r'cell\s+(\d+)',  # Cell without file
        ]
        
        for pattern in cell_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    if 'in|of|from' in pattern:
                        cell_num, file_ref = groups
                    else:
                        file_ref, cell_num = groups
                    file_ref = file_ref.replace('\\', '/')
                else:
                    cell_num = groups[0]
                    file_ref = extracted_file_path
                
                return {
                    "query_type": "cell_specific",
                    "file_path": file_ref if file_ref and '/' in file_ref else None,
                    "item_name": None,
                    "line_start": None,
                    "line_end": None,
                    "cell_number": int(cell_num),
                    "intent": f"Explain cell {cell_num}" + (f" in {file_ref}" if file_ref else "")
                }
        
        # Function/Class patterns
        func_class_patterns = [
            r'function\s+(\w+)(?:\s+(?:in|from)\s+([a-zA-Z0-9_\-/\\\.]*\.(py|ipynb)))?',
            r'class\s+(\w+)(?:\s+(?:in|from)\s+([a-zA-Z0-9_\-/\\\.]*\.(py|ipynb)))?',
            r'def\s+(\w+)(?:\s+(?:in|from)\s+([a-zA-Z0-9_\-/\\\.]*\.(py|ipynb)))?',
            r'(\w+)\s+function(?:\s+(?:in|from)\s+([a-zA-Z0-9_\-/\\\.]*\.(py|ipynb)))?',
            r'(\w+)\s+class(?:\s+(?:in|from)\s+([a-zA-Z0-9_\-/\\\.]*\.(py|ipynb)))?'
        ]
        
        for pattern in func_class_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                item_name = match.group(1)
                file_ref = match.group(2).replace('\\', '/') if len(match.groups()) > 1 and match.group(2) else extracted_file_path
                query_type = "class_specific" if "class" in pattern else "function_specific"
                
                return {
                    "query_type": query_type,
                    "file_path": file_ref if file_ref and '/' in file_ref else None,
                    "item_name": item_name,
                    "line_start": None,
                    "line_end": None,
                    "cell_number": None,
                    "intent": f"Explain {query_type.split('_')[0]} {item_name}" + (f" in {file_ref}" if file_ref else "")
                }
        
        # File-specific patterns
        if extracted_file_path and ('explain' in query_lower or 'what does' in query_lower or 'show me' in query_lower):
            return {
                "query_type": "file_specific",
                "file_path": extracted_file_path,
                "item_name": None,
                "line_start": None,
                "line_end": None,
                "cell_number": None,
                "intent": f"Explain {extracted_file_path}"
            }
        
        # Default to general semantic search
        return {
            "query_type": "general_semantic",
            "file_path": extracted_file_path,
            "item_name": None,
            "line_start": None,
            "line_end": None,
            "cell_number": None,
            "intent": "General code search"
        }

class PineconeSearchEngine:
    """Search engine for repository embeddings in Pinecone"""
    
    def __init__(self, pinecone_client, index_name: str = "baa-embeddings"):
        self.pc = pinecone_client
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        
    def search_by_embedding(self, query_embedding: List[float], limit: int = 10, 
                          file_path: Optional[str] = None, namespace: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search using vector similarity with file filtering"""
        retrieval_info = {
            "method": "vector_similarity",
            "requested_limit": limit,
            "file_path": file_path
        }
        
        try:
            filter_dict = None
            if file_path:
                # Try both exact file_path match and file_name match
                filter_dict = {
                    "$or": [
                        {"file_path": {"$eq": file_path}},
                        {"file_name": {"$eq": os.path.basename(file_path)}}
                    ]
                }
            
            results = self.index.query(
                vector=query_embedding,
                top_k=limit,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
                filter=filter_dict
            )
            
            chunks = [self._convert_match(match) for match in results.get('matches', [])]
            
            retrieval_info.update({
                "total_matches": len(chunks),
                "files_found": list(set([c.get('file_path') or c.get('file_name') for c in chunks])),
                "chunk_types": list(set([c['type'] for c in chunks]))
            })
            
            return chunks, retrieval_info
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            retrieval_info["error"] = str(e)
            return [], retrieval_info
    
    def search_by_file(self, file_path: str, namespace: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Get ALL chunks for a specific file"""
        retrieval_info = {
            "method": "file_specific",
            "file_path": file_path
        }
        
        try:
            dummy_vector = [0.0] * 768
            
            # Try multiple filter strategies
            filter_strategies = [
                {"file_path": {"$eq": file_path}},  # Exact file_path match
                {"file_name": {"$eq": os.path.basename(file_path)}},  # File name match
                {
                    "$or": [
                        {"file_path": {"$eq": file_path}},
                        {"file_name": {"$eq": os.path.basename(file_path)}}
                    ]
                }
            ]
            
            chunks = []
            for i, filter_dict in enumerate(filter_strategies):
                print(f"🔍 Trying filter strategy {i+1}: {filter_dict}")
                
                results = self.index.query(
                    vector=dummy_vector,
                    top_k=10000,  # Large limit to get all chunks
                    namespace=namespace,
                    include_metadata=True,
                    include_values=False,
                    filter=filter_dict
                )
                
                strategy_chunks = [self._convert_match(match) for match in results.get('matches', [])]
                if strategy_chunks:
                    chunks = strategy_chunks
                    retrieval_info["successful_filter"] = filter_dict
                    break
            
            # Sort by line numbers
            chunks.sort(key=lambda x: x.get('global_start_line', 0))
            
            retrieval_info.update({
                "total_chunks_found": len(chunks),
                "chunk_types": list(set([c['type'] for c in chunks])),
                "line_range": {
                    "start": chunks[0]['global_start_line'] if chunks else None,
                    "end": chunks[-1]['global_end_line'] if chunks else None
                }
            })
            
            return chunks, retrieval_info
            
        except Exception as e:
            print(f"Error searching by file: {e}")
            retrieval_info["error"] = str(e)
            return [], retrieval_info
    
    def search_by_line_range(self, start_line: int, end_line: Optional[int] = None,
                           file_path: str = None, namespace: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for chunks containing specific lines"""
        if end_line is None:
            end_line = start_line
            
        retrieval_info = {
            "method": "line_range",
            "file_path": file_path,
            "requested_lines": f"{start_line}-{end_line}"
        }
        
        try:
            # Get all chunks for the file first
            file_chunks, _ = self.search_by_file(file_path, namespace=namespace)
            
            # Filter chunks that overlap with requested range
            results = []
            for chunk in file_chunks:
                chunk_start = chunk.get('global_start_line', 0)
                chunk_end = chunk.get('global_end_line', 0)
                
                if chunk_start <= end_line and chunk_end >= start_line:
                    results.append(chunk)
            
            retrieval_info.update({
                "total_file_chunks": len(file_chunks),
                "matching_chunks": len(results)
            })
            
            return results, retrieval_info
            
        except Exception as e:
            print(f"Error searching by line range: {e}")
            retrieval_info["error"] = str(e)
            return [], retrieval_info
    
    def search_by_item(self, item_name: str, item_type: str = None,
                      file_path: Optional[str] = None, namespace: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for specific function or class"""
        retrieval_info = {
            "method": "item_specific",
            "item_name": item_name,
            "item_type": item_type,
            "file_path": file_path
        }
        
        try:
            dummy_vector = [0.0] * 768
            
            # Build filter
            filter_dict = {"item_name": {"$eq": item_name}}
            
            # Add type filter if specified
            if item_type:
                filter_dict["type"] = {"$eq": item_type}
            else:
                filter_dict["type"] = {"$in": ["function", "class"]}
            
            # Add file filter if specified
            if file_path:
                filter_dict.update({
                    "$or": [
                        {"file_path": {"$eq": file_path}},
                        {"file_name": {"$eq": os.path.basename(file_path)}}
                    ]
                })
            
            print(f"🔍 Searching with filter: {filter_dict}")
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=100,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
                filter=filter_dict
            )
            
            chunks = [self._convert_match(match) for match in results.get('matches', [])]
            
            retrieval_info.update({
                "chunks_found": len(chunks),
                "files_containing_item": list(set([c.get('file_path') or c.get('file_name') for c in chunks])),
                "item_types": list(set([c['type'] for c in chunks]))
            })
            
            return chunks, retrieval_info
            
        except Exception as e:
            print(f"Error searching by item: {e}")
            retrieval_info["error"] = str(e)
            return [], retrieval_info
    
    def search_by_cell(self, cell_number: int, file_path: Optional[str] = None,
                      namespace: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for specific notebook cell"""
        retrieval_info = {
            "method": "cell_specific",
            "cell_number": cell_number,
            "file_path": file_path
        }
        
        try:
            dummy_vector = [0.0] * 768
            
            filter_dict = {"cell_number": {"$eq": cell_number}}
            
            # Add file filter if specified
            if file_path:
                filter_dict.update({
                    "$or": [
                        {"file_path": {"$eq": file_path}},
                        {"file_name": {"$eq": os.path.basename(file_path)}}
                    ]
                })
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=100,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
                filter=filter_dict
            )
            
            chunks = [self._convert_match(match) for match in results.get('matches', [])]
            
            retrieval_info.update({
                "chunks_found": len(chunks),
                "files_containing_cell": list(set([c.get('file_path') or c.get('file_name') for c in chunks]))
            })
            
            return chunks, retrieval_info
            
        except Exception as e:
            print(f"Error searching by cell: {e}")
            retrieval_info["error"] = str(e)
            return [], retrieval_info
    
    def _convert_match(self, match) -> Dict[str, Any]:
        """Convert Pinecone match to standard format"""
        metadata = match.get('metadata', {})
        
        # Handle cell_number conversion
        cell_number = metadata.get('cell_number')
        if cell_number is not None:
            try:
                cell_number = int(float(cell_number)) if cell_number != '' else None
            except (ValueError, TypeError):
                cell_number = None
        
        return {
            "id": match.get('id', ''),
            "file_name": metadata.get('file_name', ''),
            "file_path": metadata.get('file_path', ''),
            "code_chunk": metadata.get('code_chunk', ''),
            "type": metadata.get('type', ''),
            "item_name": metadata.get('item_name', ''),
            "global_start_line": int(metadata.get('global_start_line', 0)),
            "global_end_line": int(metadata.get('global_end_line', 0)),
            "local_start_line": int(metadata.get('local_start_line', 0)),
            "local_end_line": int(metadata.get('local_end_line', 0)),
            "cell_number": cell_number,
            "score": float(match.get('score', 0.0))
        }

class RepositoryChatbot:
    """LangGraph-based repository chatbot with structured workflow"""
    
    def __init__(self, pinecone_client, gemini_api_key: str, index_name: str = "baa-embeddings"):
        self.pc = pinecone_client
        self.gemini_api_key = gemini_api_key
        self.jina_api_key = os.getenv('JINA_API_KEY')
        self.index_name = index_name
        self.query_processor = RepoQueryProcessor(gemini_api_key)
        self.search_engine = PineconeSearchEngine(pinecone_client, index_name)
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        graph = StateGraph(ChatbotState)
        
        # Add nodes
        graph.add_node("analyze_query", self._analyze_query_node)
        graph.add_node("retrieve_chunks", self._retrieve_chunks_node)
        graph.add_node("generate_response", self._generate_response_node)
        graph.add_node("handle_error", self._handle_error_node)
        
        # Define edges
        graph.add_edge(START, "analyze_query")
        graph.add_conditional_edges(
            "analyze_query",
            self._should_continue_after_analysis,
            {
                "continue": "retrieve_chunks",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "retrieve_chunks",
            self._should_continue_after_retrieval,
            {
                "continue": "generate_response",
                "error": "handle_error"
            }
        )
        graph.add_edge("generate_response", END)
        graph.add_edge("handle_error", END)
        
        return graph.compile()
    
    def _analyze_query_node(self, state: ChatbotState) -> ChatbotState:
        """Node to analyze query intent"""
        try:
            print(f"🔍 Analyzing query: {state['query']}")
            intent_analysis = self.query_processor.analyze_query_intent(state['query'])
            print(f"📊 Query analysis: {intent_analysis}")
            
            state["intent_analysis"] = intent_analysis
            state["query_type"] = intent_analysis.get("query_type", "general_semantic")
            
            return state
        except Exception as e:
            state["error"] = f"Error analyzing query: {str(e)}"
            return state
    
    def _retrieve_chunks_node(self, state: ChatbotState) -> ChatbotState:
        """Node to retrieve relevant code chunks"""
        try:
            intent_analysis = state["intent_analysis"]
            query = state["query"]
            namespace = state.get("namespace", "")
            
            # Retrieve chunks based on query type
            chunks, retrieval_info = self._retrieve_chunks_by_type(
                query, intent_analysis, namespace
            )
            
            state["retrieved_chunks"] = chunks
            state["retrieval_info"] = retrieval_info
            
            # Print retrieval summary
            self._print_retrieval_summary(chunks, retrieval_info)
            
            return state
        except Exception as e:
            state["error"] = f"Error retrieving chunks: {str(e)}"
            return state
    
    def _generate_response_node(self, state: ChatbotState) -> ChatbotState:
        """Node to generate final response"""
        try:
            query = state["query"]
            chunks = state["retrieved_chunks"]
            
            if not chunks:
                file_info = f" for file '{state['intent_analysis'].get('file_path')}'" if state['intent_analysis'].get('file_path') else ""
                state["final_response"] = f"I couldn't find any relevant code{file_info}. Please check the file path or try a different question."
                return state
            
            # Generate response using Gemini
            response = self._generate_gemini_response(query, chunks)
            state["final_response"] = response
            
            # Add to message history
            state["messages"].append(AIMessage(content=response))
            
            return state
        except Exception as e:
            state["error"] = f"Error generating response: {str(e)}"
            return state
    
    def _handle_error_node(self, state: ChatbotState) -> ChatbotState:
        """Node to handle errors"""
        error_msg = state.get("error", "Unknown error occurred")
        state["final_response"] = f"I encountered an error: {error_msg}"
        state["messages"].append(AIMessage(content=state["final_response"]))
        return state
    
    def _should_continue_after_analysis(self, state: ChatbotState) -> str:
        """Conditional edge after query analysis"""
        return "error" if state.get("error") else "continue"
    
    def _should_continue_after_retrieval(self, state: ChatbotState) -> str:
        """Conditional edge after chunk retrieval"""
        return "error" if state.get("error") else "continue"
    
    def _retrieve_chunks_by_type(self, query: str, intent_analysis: Dict[str, Any], 
                                namespace: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieve relevant code chunks based on query intent with reranking for semantic search"""
        query_type = intent_analysis.get("query_type", "general_semantic")
        file_path = intent_analysis.get("file_path")
        
        if query_type == "file_specific":
            chunks, retrieval_info = self.search_engine.search_by_file(file_path, namespace=namespace)
        
        elif query_type in ["line_specific", "line_range"]:
            start_line = intent_analysis.get("line_start")
            end_line = intent_analysis.get("line_end")
            chunks, retrieval_info = self.search_engine.search_by_line_range(
                start_line, end_line, file_path=file_path, namespace=namespace
            )
        
        elif query_type in ["function_specific", "class_specific"]:
            item_name = intent_analysis.get("item_name")
            item_type = "class" if query_type == "class_specific" else "function"
            chunks, retrieval_info = self.search_engine.search_by_item(
                item_name, item_type, file_path=file_path, namespace=namespace
            )
        
        elif query_type == "cell_specific":
            cell_number = intent_analysis.get("cell_number")
            chunks, retrieval_info = self.search_engine.search_by_cell(
                cell_number, file_path=file_path, namespace=namespace
            )
        
        else:
            # General semantic search with reranking
            query_embedding = embed_query_with_jina(query, self.jina_api_key, expected_dim=768)
            if query_embedding:
                # Get top 25 chunks for reranking
                chunks, retrieval_info = self.search_engine.search_by_embedding(
                    query_embedding, limit=25, file_path=file_path, namespace=namespace
                )
                
                # Apply reranking to get top 10 chunks
                if chunks:
                    print(f"🔄 Reranking {len(chunks)} chunks...")
                    chunks = rerank_semantic_search_results(
                        query=query,
                        chunks=chunks,
                        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
                        top_k=10
                    )
                    
                    # Update retrieval info
                    retrieval_info["reranked"] = True
                    retrieval_info["final_chunks"] = len(chunks)
                    print(f"✅ Reranked to {len(chunks)} most relevant chunks")
                else:
                    retrieval_info["reranked"] = False
            else:
                chunks, retrieval_info = [], {"error": "Failed to generate embeddings"}
        
        return chunks, retrieval_info
    
    def _print_retrieval_summary(self, chunks: List[Dict[str, Any]], retrieval_info: Dict[str, Any]) -> None:
        """Print concise retrieval information"""
        print(f"\n📊 Retrieved {len(chunks)} chunks using {retrieval_info.get('method', 'unknown')} method")
        
        if chunks:
            # Group by file
            file_groups = {}
            for chunk in chunks:
                file_key = chunk.get('file_path') or chunk.get('file_name') or 'Unknown'
                if file_key not in file_groups:
                    file_groups[file_key] = []
                file_groups[file_key].append(chunk)
            
            for file_key, file_chunks in file_groups.items():
                print(f"   📁 {file_key}: {len(file_chunks)} chunks")
                for chunk in file_chunks[:2]:  # Show first 2 chunks
                    cell_info = f" (Cell {chunk['cell_number']})" if chunk.get('cell_number') is not None else ""
                    print(f"      - Lines {chunk['global_start_line']}-{chunk['global_end_line']}: {chunk['type']} {chunk.get('item_name', 'N/A')}{cell_info}")
                if len(file_chunks) > 2:
                    print(f"      ... and {len(file_chunks) - 2} more")
    
    def _generate_gemini_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini with retrieved context"""
        
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(chunks[:8]):  # Limit to 8 chunks
            file_reference = chunk.get('file_path') or chunk.get('file_name', 'Unknown')
            context_part = f"""
## Code Chunk {i+1}
**File:** {file_reference}
**Type:** {chunk['type']} - {chunk.get('item_name', 'N/A')}
**Lines:** {chunk['global_start_line']}-{chunk['global_end_line']}
{f"**Cell:** {chunk['cell_number']}" if chunk.get('cell_number') else ""}

```python
{chunk['code_chunk']}
```
"""
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are an expert code analyst. Analyze the provided code and answer the user's question clearly and comprehensively.

## User Query
{query}

## Retrieved Code Context
{context}

## Instructions
- Provide a clear, comprehensive response
- Reference specific parts of the code when relevant
- Include line numbers, function names, or file paths when applicable
- Break down complex logic step by step
- Keep explanations focused and practical
- If the query mentioned a specific file path, prioritize explaining code from that file

## Response
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while generating the response. Please try again."
    
    def chat(self, query: str, namespace: str = "") -> str:
        """Main chat interface using LangGraph workflow"""
        try:
            # Initialize state
            initial_state = ChatbotState(
                messages=[HumanMessage(content=query)],
                query=query,
                intent_analysis={},
                retrieved_chunks=[],
                retrieval_info={},
                query_type="",
                namespace=namespace,
                error=None,
                final_response=None
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return final_state["final_response"]
            
        except Exception as e:
            print(f"Error in chat workflow: {e}")
            return f"I encountered an error: {str(e)}"
    
    def get_conversation_history(self, query: str, namespace: str = "") -> List[BaseMessage]:
        """Get full conversation history after processing a query"""
        try:
            # Initialize state
            initial_state = ChatbotState(
                messages=[HumanMessage(content=query)],
                query=query,
                intent_analysis={},
                retrieved_chunks=[],
                retrieval_info={},
                query_type="",
                namespace=namespace,
                error=None,
                final_response=None
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return final_state["messages"]
            
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return [HumanMessage(content=query), AIMessage(content=f"Error: {str(e)}")]

# Helper functions
def create_chatbot(pinecone_api_key: Optional[str] = None, 
                  gemini_api_key: Optional[str] = None,
                  index_name: str = "baa-embeddings") -> RepositoryChatbot:
    """Create and initialize the LangGraph repository chatbot"""
    
    # Get API keys from environment if not provided
    if not pinecone_api_key:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not gemini_api_key:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not pinecone_api_key or not gemini_api_key:
        raise ValueError("Both Pinecone and Gemini API keys are required")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        raise ValueError(f"Index '{index_name}' not found")
    
    print(f"✅ Connected to Pinecone index: {index_name}")
    
    return RepositoryChatbot(pc, gemini_api_key, index_name)

def initialize_pinecone_chatbot(
    pinecone_api_key: str,
    gemini_api_key: str,
    index_name: str = "baa-embeddings",
    dimension: int = 768,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1"
) -> RepositoryChatbot:
    """Initialize Pinecone connection and create LangGraph chatbot"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        print(f"⚠️ Index '{index_name}' does not exist. Please create it first using the pinecone_utils.py script.")
        raise ValueError(f"Index '{index_name}' not found")
    
    print(f"✅ Connected to Pinecone index: {index_name}")
    
    # Create chatbot
    return RepositoryChatbot(pc, gemini_api_key, index_name)

def interactive_chat_session(chatbot: RepositoryChatbot, namespace: str = ""):
    """Start an interactive chat session"""
    print("🤖 Repository Chatbot Ready!")
    print("🔄 Enhanced with LangGraph workflow management!")
    print("💡 Enhanced with file path support!")
    print("   Examples:")
    print("   - 'Explain folder/subfolder/file.py'")
    print("   - 'Show me the forward function in models/neural_network.py'")
    print("   - 'What does line 25-30 in utils/data_loader.py do?'")
    print("   - 'Explain cell 3 in notebooks/training.ipynb'")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'history' to see conversation history")
    print("=" * 60)
    
    conversation_history = []
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'history':
                print("\n📜 Conversation History:")
                for i, msg in enumerate(conversation_history, 1):
                    msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                    print(f"{i}. {msg_type}: {msg.content[:100]}...")
                continue
            
            if not query:
                continue
            
            print("\n🤖 Bot: ", end="")
            response = chatbot.chat(query, namespace)
            print(response)
            
            # Update conversation history
            messages = chatbot.get_conversation_history(query, namespace)
            conversation_history.extend(messages)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

class ChatbotWithMemory(RepositoryChatbot):
    """Extended chatbot with conversation memory"""
    
    def __init__(self, pinecone_client, gemini_api_key: str, index_name: str = "baa-embeddings"):
        super().__init__(pinecone_client, gemini_api_key, index_name)
        self.conversation_memory = []
    
    def chat_with_memory(self, query: str, namespace: str = "") -> str:
        """Chat with conversation memory"""
        # Add current query to memory
        self.conversation_memory.append(HumanMessage(content=query))
        
        # Run the workflow
        response = self.chat(query, namespace)
        
        # Add response to memory
        self.conversation_memory.append(AIMessage(content=response))
        
        return response
    
    def get_memory(self) -> List[BaseMessage]:
        """Get conversation memory"""
        return self.conversation_memory.copy()
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory = []
    
    def _generate_gemini_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini with retrieved context and conversation memory"""
        
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(chunks[:8]):  # Limit to 8 chunks
            file_reference = chunk.get('file_path') or chunk.get('file_name', 'Unknown')
            context_part = f"""
## Code Chunk {i+1}
**File:** {file_reference}
**Type:** {chunk['type']} - {chunk.get('item_name', 'N/A')}
**Lines:** {chunk['global_start_line']}-{chunk['global_end_line']}
{f"**Cell:** {chunk['cell_number']}" if chunk.get('cell_number') else ""}

```python
{chunk['code_chunk']}
```
"""
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        # Prepare conversation context
        memory_context = ""
        if self.conversation_memory:
            memory_parts = []
            for i, msg in enumerate(self.conversation_memory[-6:]):  # Last 3 exchanges
                msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                memory_parts.append(f"{msg_type}: {msg.content}")
            memory_context = f"""
## Previous Conversation Context
{chr(10).join(memory_parts)}
"""
        
        # Create prompt
        prompt = f"""You are an expert code analyst with conversation memory. Analyze the provided code and answer the user's question clearly and comprehensively, considering the conversation context.

{memory_context}

## Current User Query
{query}

## Retrieved Code Context
{context}

## Instructions
- Provide a clear, comprehensive response
- Consider previous conversation context when relevant
- Reference specific parts of the code when relevant
- Include line numbers, function names, or file paths when applicable
- Break down complex logic step by step
- Keep explanations focused and practical
- If the query mentioned a specific file path, prioritize explaining code from that file

## Response
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while generating the response. Please try again."

def demo_file_path_queries():
    """Demonstrate LangGraph chatbot capabilities"""
    demo_queries = [
        "Explain 02_Coding_Our_First_Neurons/01_A_Single_Neuron/main.py",
        "Show me the forward function in models/neural_network.py",
        "What does line 25-30 in utils/data_loader.py do?",
        "Explain cell 3 in notebooks/training.ipynb",
        "Show me the NeuralNetwork class in src/models/network.py",
        "What happens in lines 10-20 of data/preprocessing.py?",
        "Explain the train function",  # General query without file path
        "Show me utils/helpers.py line 5",
        "notebooks/analysis.ipynb cell 2 explanation"
    ]
    
    print("🎯 Demo File Path Query Patterns:")
    print("=" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"{i}. {query}")
    
    print("\n💡 The enhanced chatbot provides:")
    print("   ✅ Structured workflow with error handling")
    print("   ✅ State management across processing steps")
    print("   ✅ Conditional routing based on query analysis")
    print("   ✅ Conversation memory (with memory variant)")
    print("   ✅ Detailed logging and debugging")
    print("   ✅ Modular and extensible architecture")

# Example usage
if __name__ == "__main__":
    try:
        # Show demo patterns
        demo_langgraph_queries()
        print("\n" + "=" * 60 + "\n")
        
        # Create chatbot instance
        chatbot = create_chatbot()
        
        # Start interactive session
        interactive_chat_session(chatbot)
        
    except ValueError as e:
        print(f"❌ Setup Error: {e}")
        print("Please ensure your API keys are properly configured.")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
