import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
import re
import ast
import json
from typing import List, Tuple
import io
import sys
import time


def get_function_and_class_lines(source: str, global_line_offset: int = 0) -> List[Tuple[str, str, int, int, int, int]]:
    """
    Parse Python source code and return a list of tuples containing name, type (function/class/non-function),
    local start/end line, and global start/end line for each function, class, and non-function/class block.
    Includes all lines (empty and comments) for accurate cell processing in .ipynb files.

    Args:
        source (str): Python source code
        global_line_offset (int): Starting global line number for this source

    Returns:
        List[Tuple[str, str, int, int, int, int]]: List of (name, type, local_start, local_end, global_start, global_end)
        
    Raises:
        SyntaxError: If the source contains invalid Python syntax
    """
    try:
        lines = source.splitlines()
        # Track all lines, including empty and comments
        all_lines = list(range(1, len(lines) + 1))
        
        tree = ast.parse(source)
        items = []
        
        # Collect functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                item_type = 'function'
                local_start = node.lineno
                local_end = node.end_lineno if hasattr(node, 'end_lineno') else local_start
                
                if local_end == local_start and node.body:
                    local_end = max((n.end_lineno if hasattr(n, 'end_lineno') else n.lineno 
                                    for n in node.body), default=local_start)
                
                global_start = global_line_offset + local_start
                global_end = global_line_offset + local_end
                items.append((name, item_type, local_start, local_end, global_start, global_end))
            
            elif isinstance(node, ast.ClassDef):
                name = node.name
                item_type = 'class'
                local_start = node.lineno
                local_end = node.end_lineno if hasattr(node, 'end_lineno') else local_start
                
                if local_end == local_start and node.body:
                    local_end = max((n.end_lineno if hasattr(n, 'end_lineno') else n.lineno 
                                    for n in node.body), default=local_start)
                
                global_start = global_line_offset + local_start
                global_end = global_line_offset + local_end
                items.append((name, item_type, local_start, local_end, global_start, global_end))
        
        items.sort(key=lambda x: x[2])
        
        # Identify non-function/class blocks, including all lines
        non_function_blocks = []
        block_count = 1
        current_line = 1
        
        for name, item_type, local_start, local_end, _, _ in items + [(None, None, len(lines) + 1, len(lines) + 1, 0, 0)]:
            # Include all lines in the block, not just non-empty
            block_lines = [i for i in all_lines if current_line <= i < local_start]
            if block_lines:
                block_local_start = block_lines[0]
                block_local_end = block_lines[-1]
                block_global_start = global_line_offset + block_local_start
                block_global_end = global_line_offset + block_local_end
                non_function_blocks.append((f"block_{block_count}", 
                                           'non-function_block', 
                                           block_local_start, 
                                           block_local_end, 
                                           block_global_start, 
                                           block_global_end))
                block_count += 1
            current_line = local_end + 1
        
        # Add a final block if there are remaining lines
        if current_line <= len(lines):
            block_lines = [i for i in all_lines if current_line <= i <= len(lines)]
            if block_lines:
                block_local_start = block_lines[0]
                block_local_end = block_lines[-1]
                block_global_start = global_line_offset + block_local_start
                block_global_end = global_line_offset + block_local_end
                non_function_blocks.append((f"block_{block_count}", 
                                           'non-function_block', 
                                           block_local_start, 
                                           block_local_end, 
                                           block_global_start, 
                                           block_global_end))
        
        items.extend(non_function_blocks)
        items.sort(key=lambda x: x[2])
        
        return items
    
    except SyntaxError as e:
        raise SyntaxError(f"Invalid Python syntax: {str(e)}")
class GitHubRepoExplorer:
    def __init__(self):
        self.repo_path = None
        self.folder_structure = defaultdict(list)
        self.all_folders = []
        self.valid_extensions = {'.py', '.cpp', '.ipynb'}

    def clone_repository(self, repo_url):
        """Clone the GitHub repository or specific folder to a temporary directory"""
        try:
            repo_pattern = r'https://github\.com/([^/]+)/([^/]+)(?:/tree/[^/]+)?(/.*)?$'
            match = re.match(repo_pattern, repo_url)
            if not match:
                print("❌ Invalid GitHub URL format")
                return False
            
            owner, repo, folder_path = match.groups()
            base_repo_url = f"https://github.com/{owner}/{repo}.git"
            folder_path = folder_path.lstrip('/') if folder_path else ''
            
            temp_dir = tempfile.mkdtemp()
            print(f"Cloning repository to: {temp_dir}")
            
            process = subprocess.Popen(
                ['git', 'clone', base_repo_url, temp_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"❌ Error cloning repository: {stderr}")
                return False
            
            self.repo_path = os.path.join(temp_dir, folder_path) if folder_path else temp_dir
            
            if folder_path and not os.path.exists(self.repo_path):
                print(f"❌ Specified folder '{folder_path}' does not exist in the repository")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            print(f"✅ Repository cloned successfully! Processing: {self.repo_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error cloning repository: {e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ Git is not installed or not in PATH")
            return False
    
    def build_file_structure(self, path=None, prefix="", is_last=True):
        """Build and display the file structure, including only program files (.py, .cpp, .ipynb)"""
        if path is None:
            path = self.repo_path
            self.folder_structure.clear()
            self.all_folders.clear()
        
        path_obj = Path(path)
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
        
        try:
            items = []
            for item in path_obj.iterdir():
                if item.name.startswith('.') and item.name not in {'.gitignore', '.env.example'}:
                    continue
                if item.is_dir() and item.name in skip_dirs:
                    continue
                if item.is_file() and item.suffix not in self.valid_extensions:
                    continue
                items.append(item)
            
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1
                connector = "└── " if is_last_item else "├── "
                print(f"{prefix}{connector}{item.name}")
                
                if item.is_file():
                    folder_path = str(item.parent)
                    self.folder_structure[folder_path].append(str(item))
                elif item.is_dir():
                    self.all_folders.append(str(item))
                
                if item.is_dir():
                    extension = "    " if is_last_item else "│   "
                    self.build_file_structure(
                        item, 
                        prefix + extension, 
                        is_last_item
                    )
                    
        except PermissionError:
            print(f"{prefix}└── [Permission Denied]")
    
    def display_file_structure(self):
        """Display the complete file structure"""
        if not self.repo_path:
            print("❌ No repository or folder loaded")
            return
        
        print(f"\n📁 File Structure of: {os.path.basename(self.repo_path)}")
        print("=" * 50)
        self.build_file_structure()
        print("=" * 50)
    
    def organize_folders_with_files(self):
        """Organize folders that contain program files"""
        folders_with_files = {}
        
        root_files = self.folder_structure.get(self.repo_path, [])
        if root_files:
            folders_with_files["📁 Root Directory"] = {
                'path': self.repo_path,
                'files': root_files
            }
        
        for folder_path, files in self.folder_structure.items():
            if folder_path != self.repo_path and files:
                rel_path = os.path.relpath(folder_path, os.path.dirname(self.repo_path))
                folder_name = f"📁 {rel_path}"
                folders_with_files[folder_name] = {
                    'path': folder_path,
                    'files': files
                }
        
        return folders_with_files
    
    def list_folders_for_selection(self):
        """List all folders that contain program files with numbers for easy selection"""
        folders_with_files = self.organize_folders_with_files()
        
        if not folders_with_files:
            print("❌ No folders with program files found")
            return None
        
        print(f"\n📂 Available Folders with Program Files ({len(folders_with_files)} total):")
        print("-" * 50)
        
        folder_list = list(folders_with_files.items())
        for i, (folder_name, folder_info) in enumerate(folder_list, 1):
            file_count = len(folder_info['files'])
            print(f"{i:3d}. {folder_name} ({file_count} files)")
        
        return folder_list
    
    def list_files_in_folder(self, folder_info):
        """List all program files in the selected folder"""
        files = folder_info['files']
        folder_path = folder_info['path']
        
        print(f"\n📄 Program Files in selected folder ({len(files)} total):")
        print("-" * 40)
        
        for i, file_path in enumerate(files, 1):
            filename = os.path.basename(file_path)
            print(f"{i:3d}. {filename}")
        
        return files
    
    def display_file_contents(self, file_path):
        """Display the contents of selected program file, handling .ipynb files with cell numbers and global line numbers"""
        rel_path = os.path.relpath(file_path, os.path.dirname(self.repo_path))
        
        try:
            if file_path.endswith('.ipynb'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    notebook = json.load(file)
                
                code_cells = [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']
                total_lines = sum(len(cell.get('source', [])) for cell in code_cells)
                
                print(f"\n📖 Contents of: {rel_path}")
                print(f"📏 Total code cells: {len(code_cells)}")
                print(f"📏 Total lines in code cells: {total_lines}")
                print("=" * 60)
                
                global_line_num = 1
                for cell_num, cell in enumerate(code_cells, 1):
                    source_lines = cell.get('source', [])
                    if not source_lines:
                        print(f"Cell {cell_num}: [Empty]")
                        print("-" * 40)
                        continue
                    
                    print(f"Cell {cell_num}:")
                    print(f"{'Global':>8} | {'Cell':>6} | Code")
                    print("-" * 60)
                    for local_line_num, line in enumerate(source_lines, 1):
                        clean_line = line.rstrip('\n\r')
                        print(f"{global_line_num:>8d} | {local_line_num:>6d} | {clean_line}")
                        global_line_num += 1
                    print("-" * 40)
                
                print("=" * 60)
            
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = file.read().splitlines()
                
                print(f"\n📖 Contents of: {rel_path}")
                print(f"📏 Total lines: {len(lines)}")
                print("=" * 60)
                
                for line_num, line in enumerate(lines, 1):
                    clean_line = line.rstrip('\n\r')
                    print(f"{line_num:4d} | {clean_line}")
                
                print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    def analyze_file(self, file_path):
        """Analyze the selected file for functions, classes, and non-function/class blocks"""
        rel_path = os.path.relpath(file_path, os.path.dirname(self.repo_path))
        
        if file_path.endswith('.ipynb'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    notebook = json.load(file)
                
                code_cells = [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']
                print(f"\n🔍 Analysis of: {rel_path}")
                print(f"📋 Found {len(code_cells)} code cells")
                print("=" * 60)
                
                global_line_num = 1
                for cell_num, cell in enumerate(code_cells, 1):
                    source = ''.join(cell.get('source', []))
                    if not source.strip():
                        print(f"Cell {cell_num}: [Empty]")
                        print("-" * 40)
                        continue
                    
                    try:
                        items = get_function_and_class_lines(source, global_line_num)
                        
                        print(f"Cell {cell_num}:")
                        if items:
                            for name, item_type, local_start, local_end, global_start, global_end in items:
                                if item_type == 'non-function_block':
                                    print(f"  Non-function/class Block: {name}")
                                else:
                                    print(f"  {item_type.capitalize()}: {name}")
                                print(f"    Cell Start Line: {local_start}")
                                print(f"    Cell End Line: {local_end}")
                                print(f"    Global Start Line: {global_start}")
                                print(f"    Global End Line: {global_end}")
                        else:
                            print("  No functions or classes found")
                        print("-" * 40)
                        
                        global_line_num += len(source.splitlines())
                    
                    except SyntaxError:
                        print(f"Cell {cell_num}: [Invalid Python syntax, skipping analysis]")
                        print("-" * 40)
                
                print("=" * 60)
            
            except Exception as e:
                print(f"❌ Error analyzing file: {e}")
        
        elif file_path.endswith('.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    source = file.read()
                
                items = get_function_and_class_lines(source)
                print(f"\n🔍 Analysis of: {rel_path}")
                print(f"📋 Found {len(items)} items (functions/classes/non-function blocks, ignoring empty lines/comments)")
                print("=" * 60)
                
                for name, item_type, local_start, local_end, _, _ in items:
                    if item_type == 'non-function_block':
                        print(f"Non-function/class Block: {name}")
                    else:
                        print(f"  {item_type.capitalize()}: {name}")
                    print(f"  Start Line: {local_start}")
                    print(f"  End Line: {local_end}")
                    print("-" * 40)
                
                print("=" * 60)
            
            except (FileNotFoundError, SyntaxError) as e:
                print(f"❌ Error analyzing file: {e}")
        
        else:
            print("❌ Analysis is only supported for Python (.py) and Jupyter Notebook (.ipynb) files")
    
    def select_and_analyze_file(self):
        """Interactively select a file, allowing the user to display contents, analyze, or both"""
        folder_list = self.list_folders_for_selection()
        if not folder_list:
            print("❌ No program files available to analyze")
            return
        
        while True:
            try:
                selection = input("\nEnter the number of the folder to select (or 'q' to quit): ")
                if selection.lower() == 'q':
                    return
                folder_idx = int(selection) - 1
                if 0 <= folder_idx < len(folder_list):
                    break
                print(f"❌ Please enter a number between 1 and {len(folder_list)}")
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q' to quit")
        
        selected_folder = folder_list[folder_idx][1]
        
        files = self.list_files_in_folder(selected_folder)
        if not files:
            print("❌ No program files in the selected folder")
            return
        
        while True:
            try:
                selection = input("\nEnter the number of the file to select (or 'q' to quit): ")
                if selection.lower() == 'q':
                    return
                file_idx = int(selection) - 1
                if 0 <= file_idx < len(files):
                    break
                print(f"❌ Please enter a number between 1 and {len(files)}")
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q' to quit")
        
        selected_file = files[file_idx]
        
        while True:
            print("\n📋 Choose an action for the selected file:")
            print("  1. Display file contents")
            print("  2. Analyze functions, classes, and non-function blocks")
            print("  3. Both display contents and analyze")
            print("  q. Quit")
            action = input("\nEnter your choice (1, 2, 3, or 'q'): ")
            
            if action.lower() == 'q':
                return
            elif action == '1':
                self.display_file_contents(selected_file)
                break
            elif action == '2':
                self.analyze_file(selected_file)
                break
            elif action == '3':
                self.display_file_contents(selected_file)
                self.analyze_file(selected_file)
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 'q'")
    
    def generate_analysis_and_contents_files(self, output_dir_name):
        """Generate text files for all program files in the repository, containing analysis and contents with line numbers,
        in the user-specified directory, preserving the repository's folder structure."""
        if not self.repo_path:
            print("❌ No repository loaded")
            return
    
        # Ensure output directory is absolute
        output_dir = os.path.abspath(output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created/using output directory: {output_dir}")
    
        total_files = sum(len(files) for files in self.folder_structure.values())
        processed_files = 0
    
        for folder_path, files in self.folder_structure.items():
            # Calculate relative path from repo_path to preserve folder structure
            rel_folder = os.path.relpath(folder_path, self.repo_path)
            output_folder = os.path.join(output_dir, rel_folder)
            os.makedirs(output_folder, exist_ok=True)
        
            for file_path in files:
                processed_files += 1
                filename = os.path.basename(file_path)
                # Replace this line:
                output_filename = os.path.splitext(filename)[0] + '.txt'
                # With this:
                output_filename = filename.replace('.', '_') + '.txt'
                output_filepath = os.path.join(output_folder, output_filename)
            
                print(f"📄 Processing ({processed_files}/{total_files}): {filename}")
                
                try:
                    # Capture analysis output
                    analysis_output = io.StringIO()
                    sys.stdout = analysis_output
                    self.analyze_file(file_path)
                    sys.stdout = sys.__stdout__
                    analysis_text = analysis_output.getvalue()
                    analysis_output.close()
                    
                    # Capture contents output
                    contents_output = io.StringIO()
                    sys.stdout = contents_output
                    self.display_file_contents(file_path)
                    sys.stdout = sys.__stdout__
                    contents_text = contents_output.getvalue()
                    contents_output.close()
                    
                    # Write to text file
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Analysis and Contents of: {filename}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write("Analysis Output\n")
                        f.write("-" * 80 + "\n")
                        f.write(analysis_text)
                        f.write("\nContents Output\n")
                        f.write("-" * 80 + "\n")
                        f.write(contents_text)
                    
                    print(f"✅ Generated: {output_filepath}")
                
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
        
        print(f"\n🏁 Completed processing {processed_files} files. Output files saved in: {output_dir}")

    def generate_json_analysis_files(self, output_dir_name: str) -> None:
        """
        Generate JSON files for all program files in the repository, containing analysis of functions, classes,
        and non-function/class blocks with their code chunks, preserving the repository's folder structure.
        Processes all cells in .ipynb files, including empty and non-code cells, with correct line ranges.

        Args:
            output_dir_name (str): Directory to save the JSON files
        """
        if not self.repo_path:
            print("❌ No repository loaded")
            return

        output_dir = os.path.abspath(output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created/using output directory: {output_dir}")

        total_files = sum(len(files) for files in self.folder_structure.values())
        processed_files = 0

        for folder_path, files in self.folder_structure.items():
            rel_folder = os.path.relpath(folder_path, self.repo_path)
            output_folder = os.path.join(output_dir, rel_folder)
            os.makedirs(output_folder, exist_ok=True)

            for file_path in files:
                processed_files += 1
                filename = os.path.basename(file_path)
                output_filename = filename.replace('.', '_') + '.json'
                output_filepath = os.path.join(output_folder, output_filename)

                print(f"📄 Processing ({processed_files}/{total_files}): {filename}")
                json_data = []

                try:
                    if file_path.endswith('.ipynb'):
                        with open(file_path, 'r', encoding='utf-8') as file:
                            notebook = json.load(file)

                        all_cells = notebook.get('cells', [])
                        global_line_num = 1

                        for cell_num, cell in enumerate(all_cells, 1):
                            cell_type = cell.get('cell_type', 'unknown')
                            source = ''.join(cell.get('source', []))
                            source_lines = cell.get('source', [])
                            
                            # Include cell metadata in JSON
                            cell_local_start = 1
                            cell_local_end = len(source_lines) if source_lines else 0
                            cell_global_start = global_line_num
                            cell_global_end = global_line_num + (len(source_lines) - 1) if source_lines else global_line_num
                            cell_code_chunk = source

                            # Add cell-level item (for all cell types, including empty)
                            json_data.append({
                                "id": f"{filename}_cell_{cell_num}",
                                "file_name": filename,
                                "file_path": os.path.relpath(file_path, self.repo_path).replace('\\', '/'),
                                "item_name": f"cell_{cell_num}",
                                "type": f"{cell_type}_cell",
                                "cell_number": cell_num,
                                "local_start_line": cell_local_start,
                                "local_end_line": cell_local_end,
                                "global_start_line": cell_global_start,
                                "global_end_line": cell_global_end,
                                "code_chunk": cell_code_chunk,
                                "embedding": []
                            })

                            # Analyze code cells for functions, classes, and non-function blocks
                            if cell_type == 'code' and source.strip():
                                try:
                                    items = get_function_and_class_lines(source, global_line_num)
                                    for name, item_type, local_start, local_end, global_start, global_end in items:
                                        code_chunk = '\n'.join(source_lines[local_start-1:local_end]) if local_end >= local_start else ''
                                        chunk_lines = code_chunk.splitlines()
                                        expected_lines = global_end - global_start + 1
                                        if len(chunk_lines) != expected_lines:
                                            print(f"⚠️ Warning: Code chunk '{name}' in cell {cell_num} has {len(chunk_lines)} lines, expected {expected_lines}")
                                        item_data = {
                                            "id": f"{filename}_{name}_cell_{cell_num}",
                                            "file_name": filename,
                                            "file_path": os.path.relpath(file_path, self.repo_path).replace('\\', '/'),
                                            "item_name": name,
                                            "type": item_type,
                                            "cell_number": cell_num,
                                            "local_start_line": local_start,
                                            "local_end_line": local_end,
                                            "global_start_line": global_start,
                                            "global_end_line": global_end,
                                            "code_chunk": code_chunk,
                                            "embedding": []
                                        }
                                        json_data.append(item_data)
                                except SyntaxError:
                                    item_data = {
                                        "id": f"{filename}_cell_{cell_num}_invalid",
                                        "file_name": filename,
                                        "file_path": os.path.relpath(file_path, self.repo_path).replace('\\', '/'),
                                        "item_name": f"cell_{cell_num}",
                                        "type": "invalid_syntax",
                                        "cell_number": cell_num,
                                        "local_start_line": cell_local_start,
                                        "local_end_line": cell_local_end,
                                        "global_start_line": cell_global_start,
                                        "global_end_line": cell_global_end,
                                        "code_chunk": cell_code_chunk,
                                        "embedding": []
                                    }
                                    json_data.append(item_data)
                            
                            global_line_num += len(source_lines) if source_lines else 1

                    elif file_path.endswith('.py'):
                        with open(file_path, 'r', encoding='utf-8') as file:
                            source = file.read()
                            source_lines = source.splitlines()

                        try:
                            items = get_function_and_class_lines(source)
                            for name, item_type, local_start, local_end, global_start, global_end in items:
                                code_chunk = '\n'.join(source_lines[local_start-1:local_end]) if local_end >= local_start else ''
                                chunk_lines = code_chunk.splitlines()
                                expected_lines = global_end - global_start + 1
                                if len(chunk_lines) != expected_lines:
                                    print(f"⚠️ Warning: Code chunk '{name}' has {len(chunk_lines)} lines, expected {expected_lines}")
                                item_data = {
                                    "id": f"{filename}_{name}",
                                    "file_name": filename,
                                    "file_path": os.path.relpath(file_path, self.repo_path).replace('\\', '/'),
                                    "item_name": name,
                                    "type": item_type,
                                    "local_start_line": local_start,
                                    "local_end_line": local_end,
                                    "global_start_line": global_start,
                                    "global_end_line": global_end,
                                    "code_chunk": code_chunk,
                                    "embedding": []
                                }
                                json_data.append(item_data)
                        except SyntaxError:
                            code_chunk = source
                            item_data = {
                                "id": f"{filename}_invalid",
                                "file_name": filename,
                                "file_path": os.path.relpath(file_path, self.repo_path).replace('\\', '/'),
                                "item_name": "file_content",
                                "type": "invalid_syntax",
                                "local_start_line": 1,
                                "local_end_line": len(source_lines),
                                "global_start_line": 1,
                                "global_end_line": len(source_lines),
                                "code_chunk": code_chunk,
                                "embedding": []
                            }
                            json_data.append(item_data)

                    else:
                        print(f"❌ Skipping {filename}: Only .py and .ipynb files are supported")
                        continue

                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2)
                    print(f"✅ Generated: {output_filepath}")

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

        print(f"\n🏁 Completed processing {processed_files} files. JSON files saved in: {output_dir}")


    


    def cleanup(self):
        """
        Clean up temporary directory using Windows rmdir command with retries and fallback renaming.
        """
        if not self.repo_path or not os.path.exists(os.path.dirname(self.repo_path)):
            print("🧹 No temporary directory to clean up")
            return

        temp_dir = os.path.dirname(self.repo_path)
        max_attempts = 10
        retry_delay = 5

        for attempt in range(max_attempts):
            try:
                # Use Windows rmdir /S /Q to force delete directory
                subprocess.run(
                    ['cmd', '/C', 'rmdir', '/S', '/Q', temp_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"🧹 Cleaned up temporary directory: {temp_dir}")
                return

            except subprocess.CalledProcessError as e:
                if attempt < max_attempts - 1:
                    print(f"⚠️ Failed to delete on attempt {attempt + 1}: {e.stderr}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # Fallback: Rename directory to mark for later deletion
                    try:
                        new_dir = f"{temp_dir}_to_delete_{int(time.time())}"
                        os.rename(temp_dir, new_dir)
                        print(f"❌ Failed to delete after {max_attempts} attempts. Renamed to: {new_dir}")
                        print(f"📁 Please manually delete: {new_dir}")
                    except OSError as rename_err:
                        print(f"❌ Failed to delete or rename after {max_attempts} attempts: {e.stderr}")
                        print(f"❌ Rename error: {rename_err}")
                        print(f"📁 Please manually delete: {temp_dir}")
                    return
            except Exception as e:
                print(f"❌ Unexpected error during cleanup: {e}")
                print(f"📁 Please manually delete: {temp_dir}")
                return
