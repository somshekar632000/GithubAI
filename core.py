import os
import sys
import json
import threading
import time
from typing import Optional, Tuple, List
from datetime import datetime
import traceback
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
print("🔧 Loading environment variables...")
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"🔍 Looking for .env file at: {dotenv_path}")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print("✅ .env file found and loaded")
else:
    print("⚠️ .env file not found, trying default load...")
    load_dotenv()

# Debug: Print current working directory
print(f"📁 Current working directory: {os.getcwd()}")

def validate_api_keys():
    """Validate and debug API key loading"""
    gemini_key = os.getenv("GOOGLE_API_KEY", "").strip()
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    jina_api_key=os.getenv("JINA_API_KEY"," ").strip()
    
    print("\n🔑 API Key Validation:")
    print(f"   Gemini API Key: {'✅ Found' if gemini_key else '❌ Missing'}")
    if gemini_key:
        print(f"   Gemini Key Length: {len(gemini_key)}")
        print(f"   Gemini Key Preview: {gemini_key[:10]}...{gemini_key[-4:] if len(gemini_key) > 14 else ''}")
    
    print(f"   Pinecone API Key: {'✅ Found' if pinecone_key else '❌ Missing'}")
    if pinecone_key:
        print(f"   Pinecone Key Length: {len(pinecone_key)}")
        print(f"   Pinecone Key Preview: {pinecone_key[:10]}...{pinecone_key[-4:] if len(pinecone_key) > 14 else ''}")
    
    return gemini_key, pinecone_key

# Validate keys before proceeding
GEMINI_API_KEY, PINECONE_API_KEY = validate_api_keys()

# Import required modules
try:
    from repo_analyzer import GitHubRepoExplorer
    from gemini_embedder import embed_code_chunks_with_jina
    from pinecone_utils import initialize_pinecone, store_embeddings_in_pinecone, get_index_stats
    from pinecone_chat import create_chatbot, initialize_pinecone_chatbot
    from pinecone import Pinecone
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

class CombinedGitHubRepoChat:
    def __init__(self):
        self.processing_status = ""
        self.chatbot_instance = None
        self.is_processing = False
        self.processing_lock = threading.Lock()
        self.chat_ready = False
        self.current_output_dir = None
        self.existing_indexes = []
        
        # Get API keys from environment variables
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.jina_api_key=os.getenv("JINA_API_KEY")
        
        if not self.gemini_api_key:
            print("⚠️ Warning: GOOGLE_API_KEY environment variable not set")
        if not self.pinecone_api_key:
            print("⚠️ Warning: PINECONE_API_KEY environment variable not set")
        
    def list_existing_indexes(self) -> List[str]:
        """List existing Pinecone indexes"""
        try:
            if not self.pinecone_api_key.strip():
                return []
            
            pc = Pinecone(api_key=self.pinecone_api_key)
            indexes = pc.list_indexes()
            
            if hasattr(indexes, 'indexes'):
                return [index.name for index in indexes.indexes]
            elif isinstance(indexes, list):
                return [index.name if hasattr(index, 'name') else str(index) for index in indexes]
            else:
                return []
                
        except Exception as e:
            print(f"Error listing indexes: {e}")
            return []
    
    def get_index_info(self, index_name: str) -> str:
        """Get information about a specific index"""
        try:
            if not self.pinecone_api_key.strip() or not index_name.strip():
                return "No information available"
            
            pc = Pinecone(api_key=self.pinecone_api_key)
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            
            total_vectors = stats.get('total_vector_count', 0)
            dimension = stats.get('dimension', 'Unknown')
            
            return f"Vectors: {total_vectors:,} | Dimension: {dimension}"
            
        except Exception as e:
            return f"Error getting info: {str(e)}"
        
    def test_api_keys(self):
        """Test API keys by making simple requests"""
        test_results = {}
        
        if self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                models = genai.list_models()
                test_results['gemini'] = "✅ Valid"
                print("✅ Gemini API key validated successfully")
            except Exception as e:
                test_results['gemini'] = f"❌ Invalid: {str(e)}"
                print(f"❌ Gemini API key validation failed: {e}")
        else:
            test_results['gemini'] = "❌ Missing"
        
        if self.pinecone_api_key:
            try:
                pc = Pinecone(api_key=self.pinecone_api_key)
                indexes = pc.list_indexes()
                test_results['pinecone'] = "✅ Valid"
                print("✅ Pinecone API key validated successfully")
            except Exception as e:
                test_results['pinecone'] = f"❌ Invalid: {str(e)}"
                print(f"❌ Pinecone API key validation failed: {e}")
        else:
            test_results['pinecone'] = "❌ Missing"
        
        return test_results
    
    def refresh_indexes(self):
        """Refresh the list of existing indexes"""
        try:
            if not self.pinecone_api_key:
                return [], "❌ Pinecone API key not found in environment variables"
            
            test_results = self.test_api_keys()
            if "❌" in test_results.get('pinecone', ''):
                return [], f"❌ API Key Error: {test_results['pinecone']}"
            
            indexes = self.list_existing_indexes()
            self.existing_indexes = indexes
            
            if indexes:
                choices = indexes
                info = f"✅ Found {len(indexes)} existing indexes: {', '.join(indexes)}"
            else:
                choices = []
                info = "No existing indexes found (or none accessible with current API key)"
            
            return choices, info
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"❌ Refresh indexes error: {traceback.format_exc()}")
            return [], error_msg
    
    def initialize_existing_index(self, index_name: str):
        """Initialize chatbot with existing index"""
        try:
            if not all([self.pinecone_api_key.strip(), self.gemini_api_key.strip(), index_name.strip()]):
                return "❌ API keys not found in environment or index name missing", True
            
            self.update_status("🔄 Initializing with existing index...")
            
            self.chatbot_instance = initialize_pinecone_chatbot(
                pinecone_api_key=self.pinecone_api_key,
                gemini_api_key=self.gemini_api_key,
                index_name=index_name
            )
            
            index_info = self.get_index_info(index_name)
            
            self.chat_ready = True
            self.update_status(f"✅ Successfully connected to existing index: {index_name}")
            self.update_status(f"📊 Index info: {index_info}")
            self.update_status("🎉 Ready for chat!")
            
            return f"✅ Connected to index: {index_name}", False
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize existing index: {str(e)}"
            self.update_status(error_msg)
            return error_msg, True
    
    def extract_repo_name_from_url(self, github_url: str) -> str:
        """Extract repository name from GitHub URL"""
        try:
            parsed = urlparse(github_url)
            path = parsed.path.strip('/')
            
            if '/' in path:
                parts = path.split('/')
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1]
                    repo = repo.replace('.git', '')
                    return f"{owner}_{repo}"
            
            repo_name = path.split('/')[-1].replace('.git', '')
            return repo_name if repo_name else "unknown_repo"
            
        except Exception as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"repo_{timestamp}"
    
    def create_unique_output_dir(self, github_url: str) -> str:
        """Create a unique output directory for each repository"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_name = self.extract_repo_name_from_url(github_url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{repo_name}_{timestamp}_output"
        output_dir = os.path.join(script_dir, "outputs", dir_name)
        os.makedirs(output_dir, exist_ok=True)
        self.update_status(f"📁 Created output directory: {dir_name}")
        return output_dir
        
    def update_status(self, message: str):
        """Update processing status with timestamp"""
        with self.processing_lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.processing_status += f"\n[{timestamp}] {message}"
            print(f"[{timestamp}] {message}")
            return self.processing_status
    
    def clear_status(self):
        """Clear the processing status"""
        with self.processing_lock:
            self.processing_status = ""
            return self.processing_status
    
    def process_repository_background(self, github_url: str, index_name: str):
        """Process repository in background thread"""
        explorer = None
        
        try:
            self.is_processing = True
            self.chat_ready = False
            self.update_status("🚀 Starting repository processing...")
            
            self.current_output_dir = self.create_unique_output_dir(github_url)
            
            self.update_status(f"📡 Cloning repository: {github_url}")
            explorer = GitHubRepoExplorer()
            
            if not explorer.clone_repository(github_url):
                self.update_status("❌ Failed to clone repository")
                return False
            
            self.update_status("✅ Repository cloned successfully")
            
            self.update_status("🔍 Building file structure...")
            explorer.build_file_structure()
            
            if not explorer.folder_structure:
                self.update_status("❌ No program files found in repository")
                return False
            
            total_files = sum(len(files) for files in explorer.folder_structure.values())
            self.update_status(f"📊 Found {total_files} program files to process")
            
            self.update_status(f"📄 Generating JSON analysis files in: {os.path.basename(self.current_output_dir)}")
            explorer.generate_json_analysis_files(self.current_output_dir)
            self.update_status("✅ JSON analysis files generated")
            
            json_files = []
            for root, _, files in os.walk(self.current_output_dir):
                for file in files:
                    if file.endswith('.json'):
                        json_files.append(os.path.join(root, file))
            
            self.update_status(f"🔍 Found {len(json_files)} JSON files for embedding")
            
            if len(json_files) == 0:
                self.update_status("❌ No JSON files found to process")
                return False
            
            self.update_status("🧠 Starting embedding generation...")
            successful_embeds = 0
            
            for i, json_file in enumerate(json_files, 1):
                self.update_status(f"📑 Processing embeddings {i}/{len(json_files)}: {os.path.basename(json_file)}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if not data:
                        self.update_status(f"⚠️ Skipping empty JSON file: {os.path.basename(json_file)}")
                        continue
                    
                    success = embed_code_chunks_with_jina(  # Changed function name
                        json_file_path=json_file,
                        api_key=self.jina_api_key,  # Changed variable name
                        batch_size=50,  # Increased batch size
                        delay_between_requests=0.2  # Reduced delay
                    )
                    
                    if success:
                        successful_embeds += 1
                        
                except Exception as e:
                    self.update_status(f"❌ Error generating embeddings for {os.path.basename(json_file)}: {str(e)}")
            
            if successful_embeds == 0:
                self.update_status("❌ No embeddings generated successfully")
                return False
            
            self.update_status(f"✅ Generated embeddings for {successful_embeds}/{len(json_files)} files")
            
            self.update_status("📦 Initializing Pinecone...")
            try:
                pinecone_client, pinecone_index = initialize_pinecone(
                    api_key=self.pinecone_api_key,
                    index_name=index_name,
                    dimension=768,
                    recreate_index=True
                )
                self.update_status("✅ Pinecone initialized")
            except Exception as e:
                self.update_status(f"❌ Failed to initialize Pinecone: {str(e)}")
                return False
            
            self.update_status("📤 Storing embeddings in Pinecone...")
            pinecone_success = 0
            
            for i, json_file in enumerate(json_files, 1):
                self.update_status(f"📤 Storing embeddings {i}/{len(json_files)}: {os.path.basename(json_file)}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    has_embeddings = any(item.get('embedding') for item in data)
                    if not has_embeddings:
                        self.update_status(f"⚠️ No embeddings found in {os.path.basename(json_file)}")
                        continue
                    
                    success = store_embeddings_in_pinecone(
                        json_file=json_file,
                        index=pinecone_index
                    )
                    
                    if success:
                        pinecone_success += 1
                        
                except Exception as e:
                    self.update_status(f"❌ Error storing embeddings for {os.path.basename(json_file)}: {str(e)}")
            
            if pinecone_success == 0:
                self.update_status("❌ No embeddings stored successfully")
                return False
            
            self.update_status(f"✅ Stored embeddings for {pinecone_success}/{len(json_files)} files")
            
            self.update_status("🤖 Initializing chatbot...")
            try:
                self.chatbot_instance = initialize_pinecone_chatbot(
                    pinecone_api_key=self.pinecone_api_key,
                    gemini_api_key=self.gemini_api_key,
                    index_name=index_name
                )
                self.update_status("✅ Chatbot initialized successfully")
                self.chat_ready = True
            except Exception as e:
                self.update_status(f"❌ Failed to initialize chatbot: {str(e)}")
                return False
            
            self.update_status("🎉 Processing completed! You can now ask questions.")
            self.update_status(f"📂 Output files saved in: {self.current_output_dir}")
            return True
            
        except Exception as e:
            self.update_status(f"❌ Fatal error: {str(e)}")
            self.update_status(f"❌ Traceback: {traceback.format_exc()}")
            return False
        finally:
            if explorer:
                try:
                    explorer.cleanup()
                    self.update_status("🧹 Cleanup completed")
                except Exception as e:
                    self.update_status(f"⚠️ Cleanup warning: {str(e)}")
            
            self.is_processing = False
    
    def process_repository(self, github_url: str, index_name: str):
        """Start repository processing"""
        if self.is_processing:
            return "⚠️ Processing already in progress.", True
        
        if not github_url.strip():
            return "❌ GitHub URL is required", True
        
        if not index_name.strip():
            return "❌ Index name is required", True
        
        if not self.gemini_api_key:
            return "❌ Gemini API key not found in environment variables", True
        
        if not self.pinecone_api_key:
            return "❌ Pinecone API key not found in environment variables", True
        
        if not github_url.strip().startswith(('https://github.com/', 'http://github.com/', 'git@github.com:')):
            return "❌ Please provide a valid GitHub URL", True
        
        self.clear_status()
        
        processing_thread = threading.Thread(
            target=self.process_repository_background,
            args=(github_url, index_name)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        return "🚀 Processing started...", False
    
    def get_processing_status(self):
        """Get current processing status"""
        with self.processing_lock:
            status = self.processing_status
            if self.is_processing:
                status += "\n\n🔄 Processing in progress..."
            elif self.chat_ready:
                status += "\n\n✅ Ready for chat!"
                if self.current_output_dir:
                    status += f"\n📂 Files saved in: {os.path.basename(self.current_output_dir)}"
            return status
    
    def chat_with_repo(self, message: str, history: List[List[str]]):
        """Chat with the processed repository"""
        if not self.chatbot_instance or not self.chat_ready:
            error_msg = "❌ Repository not processed yet. Please process a repository first or connect to an existing index."
            history.append([message, error_msg])
            return history, ""
        
        if not message.strip():
            return history, ""
        
        try:
            response = self.chatbot_instance.chat(message)
            history.append([message, response])
            return history, ""
            
        except Exception as e:
            error_response = f"❌ Error: {str(e)}"
            history.append([message, error_response])
            return history, ""
    
    def clear_chat(self):
        """Clear chat history"""
        return []