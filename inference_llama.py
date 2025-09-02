# =========================
# Imports & Constants
# =========================
import os
import re
import pickle
import argparse
import json
import logging
import subprocess
from datetime import datetime
import time
from typing import Any, List, Tuple, Dict, Optional
import torch
import numpy as np
from tqdm import tqdm
import faiss
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

from function_database import FunctionDatabase
from asmlm import AsmLMModule, TOKENIZER

# Constants
DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 32
DEFAULT_OLLAMA_MODEL = "llama3.1:8b-instruct-q4_0"
RESUME_FILE = None  # Example: 'intermediate_symbol_inference_results_36300_of_104031_20250511_082632.json' or None

# Environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =========================
# Logging Utilities
# =========================
def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Setup logging configuration
    
    Args:
        log_file (Optional[str]): Log file path
        level (int): Log level
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=log_format, handlers=handlers)

setup_logging()

# =========================
# GPU Diagnostics & Monitoring
# =========================
def check_ollama_status() -> None:
    """Check Ollama server status"""
    print("=== Ollama Server Status Check ===")
    
    try:
        # Check Ollama server status
        models = ollama.list()
        print(f"Ollama server connection successful")
        print(f"Available models count: {len(models['models'])}")
        
        # Output available models
        for i, model in enumerate(models['models']):
            # Try various possible field names
            model_name = None
            for key in ['name', 'model', 'id', 'tag', 'digest']:
                if key in model:
                    model_name = model[key]
                    break
            if not model_name:
                model_name = f"model_{i+1}"
            model_size = model.get('size', 0)
            print(f"  - {model_name} ({model_size/1024**3:.1f}GB)")
            
        # GPU information (PyTorch-based)
        if torch.cuda.is_available():
            print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            
            # GPU memory information
            memory_info = torch.cuda.get_device_properties(0)
            total_memory = memory_info.total_memory / 1024**3
            print(f"Total GPU memory: {total_memory:.1f}GB")
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Currently allocated memory: {allocated:.1f}GB")
            print(f"Currently reserved memory: {reserved:.1f}GB")
            print(f"Available memory: {total_memory - reserved:.1f}GB")
        
    except Exception as e:
        print(f"Error checking Ollama server: {e}")

def check_ollama_model(model_name: str) -> bool:
    """Check if Ollama model exists
    
    Args:
        model_name (str): Model name to check
        
    Returns:
        bool: True if model exists, False otherwise
    """
    try:
        models = ollama.list()
        available_models = []
        
        for model in models['models']:
            # Try various possible field names
            name = None
            for key in ['name', 'model', 'id', 'tag', 'digest']:
                if key in model:
                    name = model[key]
                    break
            if name:
                available_models.append(name)
        
        if model_name in available_models:
            print(f"Model '{model_name}' is available")
            return True
        else:
            print(f"Model '{model_name}' not found")
            print(f"Available models: {available_models}")
            # For debugging: output first model's complete structure
            if models['models']:
                print(f"First model structure: {models['models'][0]}")
            return False
            
    except Exception as e:
        print(f"Error checking model: {e}")
        return False

def monitor_gpu_usage() -> None:
    """Monitor GPU usage in real-time"""
    if not torch.cuda.is_available():
        return
    
    try:
        # GPU memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        
        # GPU utilization (if pynvml available)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU Utilization: {util.gpu}%")
        except ImportError:
            pass
            
    except Exception as e:
        print(f"GPU monitoring error: {e}")

# =========================
# Parsing Functions
# =========================
def parse_c_file(c_file_path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Parse function information from C file
    
    Args:
        c_file_path (str): C file path
        
    Returns:
        Tuple[List[dict], int]: (function info list, decompile failure count)
        
    Raises:
        FileNotFoundError: When file doesn't exist
    """
    functions: List[Dict[str, Any]] = []
    decompile_fail_count = 0
    
    if not os.path.exists(c_file_path):
        logging.error(f"C file does not exist: {c_file_path}")
        raise FileNotFoundError(f"C file does not exist: {c_file_path}")
    
    with open(c_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        pattern = r'/\* Function: (.*?) at (0x[0-9A-Fa-f]+) \*/(.*?)(?=/\* Function:|$)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            func_name = match.group(1).strip()
            func_addr = match.group(2).lower()
            func_addr = re.sub(r'0x0+', '0x', func_addr)
            decomp_code = match.group(3).strip()
            
            # Check for decompilation failure marker (Korean text preserved for pattern matching)
            if decomp_code.startswith("/* [!] 디컴파일"):
                decompile_fail_count += 1
                continue
                
            functions.append({
                "func_name": func_name,
                "func_addr": func_addr,
                "decomp_code": decomp_code,
                "asm_code": None
            })
    
    return functions, decompile_fail_count

# =========================
# Assembly/Embedding Functions
# =========================
def load_assembly_code(pickle_path: str) -> Dict[str, Any]:
    """Load assembly code from pickle file
    
    Args:
        pickle_path (str): Pickle file path
        
    Returns:
        dict: Assembly code dictionary
        
    Raises:
        FileNotFoundError: When file doesn't exist
        pickle.UnpicklingError: When pickle file loading fails
    """
    asm_functions: Dict[str, Any] = {}
    duplicate_count = 0
    
    if not os.path.exists(pickle_path):
        logging.error(f"Pickle file does not exist: {pickle_path}")
        raise FileNotFoundError(f"Pickle file does not exist: {pickle_path}")
    
    try:
        with open(pickle_path, 'rb') as f:
            func_list = pickle.load(f)
            logging.info(f"\nLoading pickle file: {pickle_path}")
            logging.info(f"Total functions in pickle: {len(func_list)}")
            
            for idx, func in enumerate(func_list):
                if func:
                    raw_addr = func[0][0]
                    func_addr = f"0x{raw_addr.lower().lstrip('0x')}"
                    func_addr = re.sub(r'0x0+', '0x', func_addr)
                    
                    if func_addr in asm_functions:
                        duplicate_count += 1
                        logging.warning(f"  Duplicate address found: {func_addr}")
                        continue
                        
                    asm_functions[func_addr] = func
                    
            if duplicate_count > 0:
                logging.warning(f"  Total {duplicate_count} duplicate addresses found.")
                
    except pickle.UnpicklingError as e:
        logging.error(f"Pickle file loading failed: {e}")
        raise pickle.UnpicklingError(f"Pickle file loading failed: {e}")
    
    return asm_functions

def match_functions(c_file_path: str, pickle_path: str) -> List[Dict[str, Any]]:
    """Match C file and pickle file functions
    
    Args:
        c_file_path (str): C file path
        pickle_path (str): Pickle file path
        
    Returns:
        list: Matched function list
    """
    all_functions: List[Dict[str, Any]] = []
    matched_count = 0
    total_count = 0
    unmatched_functions = []
    total_decompile_fail = 0
    duplicate_functions = set()
    
    # Parse C file
    functions, decompile_fail = parse_c_file(c_file_path)
    total_decompile_fail += decompile_fail
    total_count += len(functions)
    
    # Load assembly code
    asm_functions = load_assembly_code(pickle_path)
    
    # Match functions
    for func in functions:
        func_addr = func["func_addr"].lower()
        func_key = f"{os.path.basename(c_file_path)}:{func_addr}"
        
        if func_key in duplicate_functions:
            logging.warning(f"  Duplicate function found: {func['func_name']} ({func_addr})")
            continue
            
        duplicate_functions.add(func_key)
        
        if func_addr in asm_functions:
            func["asm_code"] = asm_functions[func_addr]
            all_functions.append(func)
            matched_count += 1
        else:
            logging.warning(f"\nMatching failed - Function: {func['func_name']}")
            logging.warning(f"  C file address: {func['func_addr']}")
            logging.warning(f"  Available addresses: {list(asm_functions.keys())[:5]}")
            unmatched_functions.append({
                "function": func["func_name"],
                "address": func_addr,
                "available_addresses": list(asm_functions.keys())[:5]
            })
    
    # Log results
    logging.info(f"\nMatching results: {matched_count}/{total_count} functions successfully matched.")
    logging.info(f"Decompilation failures: {total_decompile_fail}")
    logging.info(f"Duplicate functions: {len(duplicate_functions) - matched_count}")
    
    if unmatched_functions:
        logging.warning("\nUnmatched function samples (max 5):")
        for func in unmatched_functions[:5]:
            logging.warning(f"  - {func['function']} ({func['address']})")
            logging.warning(f"    Available addresses: {func['available_addresses']}")
    
    return all_functions

def generate_embedding(model: torch.nn.Module, asm_code: Any, device: Optional[str] = None) -> np.ndarray:
    """Generate embedding for assembly code
    
    Args:
        model: kTrans model
        asm_code: Assembly code
        device: Device (default: model's current device)
        
    Returns:
        np.ndarray: Normalized embedding vector
    """
    if device is None:
        device = next(model.parameters()).device
    
    encoding = TOKENIZER.encode_func(asm_code)
    
    with torch.no_grad():
        embedding = model(
            torch.tensor([encoding['func_token_ids']]).to(device),
            torch.tensor([encoding['func_insn_type_ids']]).to(device),
            torch.tensor([encoding['func_opnd_type_ids']]).to(device),
            torch.tensor([encoding['func_reg_id_ids']]).to(device),
            torch.tensor([encoding['func_opnd_r_ids']]).to(device),
            torch.tensor([encoding['func_opnd_w_ids']]).to(device),
            torch.tensor([encoding['func_eflags_ids']]).to(device)
        )[:,0,:].cpu().numpy()
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    
    return embedding[0]

def create_embeddings(model: torch.nn.Module, functions: List[Dict[str, Any]], 
                     batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """Create assembly code embeddings using kTrans model
    
    Args:
        model: kTrans model
        functions: Function list
        batch_size: Batch size
        
    Returns:
        tuple: (function list, embedding list)
    """
    embeddings: List[np.ndarray] = []
    device = next(model.parameters()).device
    
    for i in tqdm(range(0, len(functions), batch_size), desc="Creating embeddings"):
        batch = functions[i:i + batch_size]
        batch_embeddings = []
        
        for func in batch:
            embedding = generate_embedding(model, func["asm_code"], device)
            batch_embeddings.append(embedding)
            
        embeddings.extend(batch_embeddings)
    
    return functions, embeddings

# =========================
# Database Functions
# =========================
def build_database(c_file_path: str, pickle_path: str, model_path: str, 
                  save_dir: str, device: str = DEFAULT_DEVICE) -> Optional[FunctionDatabase]:
    """Build function database with matching and embedding generation
    
    Args:
        c_file_path: C file path
        pickle_path: Pickle file path
        model_path: Model path
        save_dir: Save directory
        device: Device
        
    Returns:
        FunctionDatabase: Function database
    """
    # Load kTrans model
    logging.info('Loading kTrans model...')
    pl_model = AsmLMModule.load_from_checkpoint(model_path, total_steps=0)
    model = pl_model.bert
    model.to(torch.device(device))
    model.eval()
    
    # Match functions
    functions = match_functions(c_file_path, pickle_path)
    logging.info(f"\nTotal functions found: {len(functions)}")
    
    # Filter functions with assembly code
    functions = [f for f in functions if f['asm_code']]
    logging.info(f"Functions with assembly code: {len(functions)}")
    
    if not functions:
        logging.warning("No functions to process. Exiting...")
        return None
    
    # Create embeddings
    unique_functions, embeddings = create_embeddings(model, functions)
    
    # Build database
    db = FunctionDatabase(save_dir)
    for func, embedding in tqdm(zip(unique_functions, embeddings), 
                               total=len(unique_functions), desc="Adding functions to database"):
        db.add_function(
            func_name=func["func_name"],
            asm_code=func["asm_code"],
            decomp_code=func["decomp_code"],
            embedding_asm=embedding
        )
    
    # Save database
    logging.info(f"Saving database to {save_dir}...")
    db.save()
    logging.info("Database build completed!")
    
    # Log statistics
    logging.info(f"Total functions in database: {len(unique_functions)}")
    logging.info(f"Total original functions: {len(functions)}")
    logging.info(f"Reduced by: {len(functions) - len(unique_functions)} functions through deduplication")
    logging.info(f"\nDatabase status:")
    logging.info(f"Function map size: {len(db.function_map)}")
    logging.info(f"FAISS index exists: {db.asm_db is not None}")
    
    return db

# =========================
# Local LLM Initialization (Optimized Version)
# =========================
def initialize_ollama_client(model_name: str = DEFAULT_OLLAMA_MODEL) -> Optional[str]:
    """Initialize Ollama client
    
    Args:
        model_name (str): Ollama model name
        
    Returns:
        str | None: Model name if successful, None if failed
    """
    print("=== Ollama Client Initialization ===")
    
    # 1. Check Ollama server status
    check_ollama_status()
    
    # 2. Check model existence
    if not check_ollama_model(model_name):
        print(f"⚠️ Model '{model_name}' not found. Please download it first using:")
        print(f"  ollama pull {model_name}")
        return None
    
    print(f"✅ Ollama client initialization completed: {model_name}")
    return model_name

# =========================
# LLM Symbol Inference
# =========================
def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in text (estimate for Ollama since exact counting is not provided)
    
    Args:
        text (str): Text to count tokens
        model_name (str): Model name (for compatibility)
        
    Returns:
        int: Estimated token count
    """
    # Ollama doesn't provide exact token counting, so we estimate
    return int(len(text.split()) * 1.3)  # Approximately 1.3x word count

def infer_symbol_with_llm(decomp_code: str, similar_functions: List[Tuple[Any, float]], 
                         model_name: str, debug: bool = False) -> Optional[Dict[str, Any]]:
    """Infer symbol using Ollama LLM
    
    Args:
        decomp_code (str): Decompiled code
        similar_functions (List[tuple]): Similar function list
        model_name (str): Ollama model name
        debug (bool): Enable debug mode
        
    Returns:
        dict | None: Inference result
    """
    if model_name is None:
        logging.error("Ollama model name is None.")
        return None
    
    try:
        # Prepare similar functions info
        similar_funcs_info = []
        for i, (func_data, score) in enumerate(similar_functions, 1):
            similar_funcs_info.append({
                "name": func_data.func_name,
                "similarity": score,
                "decomp_code": func_data.decomp_code
            })
        
        system_prompt = "You are an expert in analyzing assembly code and decompiled code to understand function meanings and infer appropriate names. Please accurately grasp the purpose and behavior of functions to suggest meaningful names. You should write the function name in snake_case format (e.g., calculate_total, process_user_data, validate_input)."
        
        user_prompt = f"""
Below are examples of decompiled functions with all symbol information removed.
All function names in the form "FUN_{{num}}" in both the declaration and body are placeholders and do not represent the original function names. Do not attempt to infer any meaning from the numbers in these names.

Your final task is to infer the original, meaningful function name for the function named FUN_0 in the query function's decompiled code, based on its decompiled code and the provided similar function examples.
Note: "FUN_0" may also appear in the similar function examples, but you must only infer the original name for FUN_0 in the query function's code.

Instructions:
1. Carefully analyze the behavior and purpose of the query function (FUN_0) using its decompiled code.
2. Compare the query function's behavior with the similar function examples provided below.
3. If you find a function among the similar examples whose behavior closely matches the query function, use its name as a reference for inferring the query function's name.
4. If there is no sufficiently similar function among the examples, infer the most appropriate name based solely on the query function's behavior.
5. All "FUN_{{num}}" names in the code (including called functions) are anonymized and do not provide semantic hints. Ignore the numbers in these names.
6. After you infer the function name, internally double-check that your proposed name is the most accurate and appropriate for the function's purpose and behavior. If not, revise it internally before outputting.

Important: Do all reasoning and checking internally. For your final answer, output only the inferred function name in snake_case.
Do not include any explanations, reasoning, or additional information.

---
Similar function examples:
{json.dumps(similar_funcs_info, indent=2)}

Query function's decompiled code:
{decomp_code}

Output:
Only output the inferred function name in snake_case.
"""
        
        # Count tokens (estimated)
        prompt_tokens = count_tokens(user_prompt, model_name)
        system_tokens = count_tokens(system_prompt, model_name)
        total_tokens = prompt_tokens + system_tokens
        
        if debug:
            monitor_gpu_usage()
        
        try:
            # Record start time
            start_time = time.time()
            
            # Inference through Ollama
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 30,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["\n", " ", "."]
                },
                stream=False
            )
            
            # Record end time and calculate elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if debug:
                print(f"⏱️ Inference time: {elapsed_time:.2f} seconds")
                monitor_gpu_usage()
            
            response_text = response['message']['content']
            inferred_name = response_text.strip().split('\n')[0].strip().split()[0].strip()
            
            if not inferred_name:
                logging.warning("LLM did not return a function name.")
                return None
                
            return {
                "inferred_name": inferred_name,
                "inference_reason": "Inference through Ollama",
                "full_response": response_text.strip(),
                "token_stats": {
                    "prompt_tokens": prompt_tokens,
                    "system_tokens": system_tokens,
                    "total_tokens": total_tokens
                },
                "time_stats": {
                    "elapsed_time": elapsed_time,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logging.error(f"\nError during Ollama inference: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"\nError during LLM symbol inference: {str(e)}")
        return None

# =========================
# Test Set Processing
# =========================
def collect_testset_functions(root_dirs: List[str]) -> List[Dict[str, Any]]:
    """Collect functions from test set directories (including subdirectories)
    
    Args:
        root_dirs: List of root directory paths
        
    Returns:
        List[Dict[str, Any]]: All collected function information
    """
    all_functions: List[Dict[str, Any]] = []
    tasks = []
    
    for root_dir in root_dirs:
        # Process files in root directory
        c_files = {f[:-2]: f for f in os.listdir(root_dir) if f.endswith('.c')}
        pkl_files = {f[:-4]: f for f in os.listdir(root_dir) if f.endswith('.pkl')}
        common = set(c_files.keys()) & set(pkl_files.keys())
        logging.info(f"[Root directory] Matching file count: {len(common)}")
        
        for binary_name in common:
            c_file_path = os.path.join(root_dir, c_files[binary_name])
            pkl_file_path = os.path.join(root_dir, pkl_files[binary_name])
            tasks.append(("root", binary_name, c_file_path, pkl_file_path))
        
        # Process subdirectories
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            c_files = {f[:-2]: f for f in os.listdir(subdir_path) if f.endswith('.c')}
            pkl_files = {f[:-4]: f for f in os.listdir(subdir_path) if f.endswith('.pkl')}
            common = set(c_files.keys()) & set(pkl_files.keys())
            logging.info(f"[{subdir}] Matching file count: {len(common)}")
            
            for binary_name in common:
                c_file_path = os.path.join(subdir_path, c_files[binary_name])
                pkl_file_path = os.path.join(subdir_path, pkl_files[binary_name])
                tasks.append((subdir, binary_name, c_file_path, pkl_file_path))

    def process_file(args):
        """Process individual file"""
        subdir, binary_name, c_file_path, pkl_file_path = args
        try:
            functions = match_functions(c_file_path, pkl_file_path)
            for func in functions:
                func["binary"] = binary_name
                func["subdir"] = subdir
            return functions
        except Exception as e:
            logging.error(f"Error collecting function info for {subdir}/{binary_name}: {e}")
            return []

    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(process_file, task): task for task in tasks}
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Collecting files in parallel"):
            result = future.result()
            if result:
                results.extend(result)
    
    return results

def save_intermediate_results(all_results: List[Dict[str, Any]], current_index: int, 
                            total_functions: int) -> str:
    """Save intermediate results
    
    Args:
        all_results: All results so far
        current_index: Current processing index
        total_functions: Total number of functions
        
    Returns:
        str: Saved file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_file = f"intermediate_symbol_inference_results_{current_index}_of_{total_functions}_{timestamp}.json"
    
    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)
    
    # Remove assembly and decompiled code before saving
    intermediate_results = []
    for result in all_results:
        result_copy = dict(result)
        
        # Remove decomp_code from function info
        if "function" in result_copy and "decomp_code" in result_copy["function"]:
            result_copy["function"] = dict(result_copy["function"])
            result_copy["function"].pop("decomp_code", None)
        
        # Remove asm_code and decomp_code from similar functions
        if "similar_functions" in result_copy:
            new_similars = []
            for sim in result_copy["similar_functions"]:
                sim_copy = dict(sim)
                sim_copy.pop("asm_code", None)
                sim_copy.pop("decomp_code", None)
                new_similars.append(sim_copy)
            result_copy["similar_functions"] = new_similars
        
        intermediate_results.append(result_copy)
    
    with open(intermediate_file, 'w', encoding='utf-8') as f:
        json.dump(intermediate_results, f, ensure_ascii=False, indent=2)
    
    return intermediate_file

def save_statistics(token_stats_list: List[Dict], time_stats_list: List[Dict], 
                   slow_count: int, fast_count: int) -> str:
    """Save token and time statistics
    
    Args:
        token_stats_list: List of token statistics
        time_stats_list: List of time statistics
        slow_count: Number of slow inferences
        fast_count: Number of fast inferences
        
    Returns:
        str: Statistics file path
    """
    if not token_stats_list or not time_stats_list:
        return ""
    
    # Calculate aggregated statistics
    total_prompt_tokens = sum(stat["prompt_tokens"] for stat in token_stats_list)
    total_tokens = sum(stat["total_tokens"] for stat in token_stats_list)
    total_elapsed_time = sum(stat["elapsed_time"] for stat in time_stats_list)
    
    avg_prompt_tokens = total_prompt_tokens / len(token_stats_list)
    avg_total_tokens = total_tokens / len(token_stats_list)
    avg_elapsed_time = total_elapsed_time / len(time_stats_list)
    
    max_prompt_tokens = max(stat["prompt_tokens"] for stat in token_stats_list)
    max_total_tokens = max(stat["total_tokens"] for stat in token_stats_list)
    min_prompt_tokens = min(stat["prompt_tokens"] for stat in token_stats_list)
    min_total_tokens = min(stat["total_tokens"] for stat in token_stats_list)
    
    max_time = max(stat["elapsed_time"] for stat in time_stats_list)
    min_time = min(stat["elapsed_time"] for stat in time_stats_list)
    
    # Log statistics
    logging.info("\n" + "="*50)
    logging.info("=== Final Performance Statistics ===")
    logging.info("="*50)
    logging.info(f"Total processed functions: {len(token_stats_list)}")
    logging.info(f"Total prompt tokens: {total_prompt_tokens:,}")
    logging.info(f"Total tokens: {total_tokens:,}")
    logging.info(f"Average prompt tokens: {avg_prompt_tokens:.0f}")
    logging.info(f"Average total tokens: {avg_total_tokens:.0f}")
    logging.info(f"Token range: {min_total_tokens:,} ~ {max_total_tokens:,}")
    
    logging.info(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/3600:.1f} hours)")
    logging.info(f"Average response time: {avg_elapsed_time:.2f} seconds")
    logging.info(f"Time range: {min_time:.2f}s ~ {max_time:.2f}s")
    logging.info(f"Requests per second: {1/avg_elapsed_time:.2f}")
    logging.info(f"Slow inferences (>60s): {slow_count}")
    logging.info(f"Fast inferences (≤60s): {fast_count}")
    
    # Save statistics to JSON file
    stats_file = f"llm_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_functions": len(token_stats_list),
                "token_stats": {
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_tokens": total_tokens,
                    "average_prompt_tokens": avg_prompt_tokens,
                    "average_total_tokens": avg_total_tokens,
                    "max_prompt_tokens": max_prompt_tokens,
                    "max_total_tokens": max_total_tokens,
                    "min_prompt_tokens": min_prompt_tokens,
                    "min_total_tokens": min_total_tokens
                },
                "time_stats": {
                    "total_elapsed_time": total_elapsed_time,
                    "average_elapsed_time": avg_elapsed_time,
                    "max_elapsed_time": max_time,
                    "min_elapsed_time": min_time,
                    "requests_per_second": 1/avg_elapsed_time,
                    "slow_inference_count": slow_count,
                    "fast_inference_count": fast_count
                }
            },
            "detailed_stats": {
                "token_stats": token_stats_list,
                "time_stats": time_stats_list
            }
        }, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\nStatistics saved to {stats_file}")
    return stats_file

# =========================
# Main Workflow
# =========================
def main() -> None:
    """Main entry point function"""
    # Configuration paths (update these paths for your environment)
    model_path = r"path/to/your/ktrans-model.ckpt"
    db_path = r"path/to/your/database"
    pickle_dir = r"path/to/your/pickle/files"
    c_dir = r"path/to/your/c/files"
    
    # Configuration parameters
    device = DEFAULT_DEVICE
    k = 10  # Number of similar functions to retrieve
    use_llm = True
    output_json = True
    
    # Validate database path
    if not os.path.exists(db_path):
        logging.error(f"Error: Database path does not exist: {db_path}")
        return

    # Collect all test set functions
    all_functions = collect_testset_functions([c_dir])
    logging.info(f"Total {len(all_functions)} function information collected.")
    
    if not all_functions:
        logging.warning("No functions to process. Exiting.")
        return

    # Load kTrans model
    logging.info('Loading kTrans model...')
    pl_model = AsmLMModule.load_from_checkpoint(model_path, total_steps=0)
    model = pl_model.bert
    model.to(torch.device(device))
    model.eval()
    
    # Load database
    db = FunctionDatabase(db_path)
    db.load()
    
    model_device = next(model.parameters()).device

    # Initialize Ollama model
    try:
        llm_model = initialize_ollama_client(model_name=DEFAULT_OLLAMA_MODEL)
        if llm_model is None:
            logging.error("Ollama model initialization failed")
            return
        logging.info("Ollama model successfully initialized.")
    except Exception as e:
        logging.error(f"Error during Ollama model initialization: {e}")
        return

    # Resume functionality: continue from intermediate save file
    all_results = []
    resume_from = 0
    
    if RESUME_FILE is not None and os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        resume_from = len(all_results)
        logging.info(f"[Resume] Restored {resume_from} results from {RESUME_FILE}, continuing.")
    else:
        logging.info("[Resume] No intermediate save file specified or found, starting from beginning.")

    total_functions = len(all_functions)
    
    # Statistics tracking
    token_stats_list = []
    time_stats_list = []
    slow_inference_count = 0
    fast_inference_count = 0
    
    # Process each function
    for i, func in enumerate(tqdm(all_functions[resume_from:], desc="Batch function inference", 
                                 initial=resume_from, total=total_functions)):
        
        # Extract function information
        func_name = func["func_name"]
        func_addr = func["func_addr"]
        asm_code = func["asm_code"]
        decomp_code = func["decomp_code"]
        binary_name = func["binary"]
        
        # Generate embedding and find similar functions
        embedding = generate_embedding(model, asm_code, model_device)
        similar_functions = db.search_similar_functions(asm_code, embedding, k=k)
        
        # Prepare result data
        result_data = {
            "binary": binary_name,
            "function": {
                "name": func_name,
                "address": func_addr,
                "decomp_code": decomp_code
            },
            "similar_functions": []
        }
        
        # Add similar functions to result
        for j, (func_data, score) in enumerate(similar_functions, 1):
            result_data["similar_functions"].append({
                "name": func_data.func_name,
                "similarity": float(score),
                "asm_code": func_data.asm_code,
                "decomp_code": func_data.decomp_code
            })
        
        # Perform LLM inference if enabled
        llm_result = None
        if use_llm:
            # Use debug mode for first function
            debug_mode = (i == 0)
            
            llm_result = infer_symbol_with_llm(
                decomp_code, 
                similar_functions, 
                llm_model, 
                debug=debug_mode
            )
            
            if llm_result:
                result_data["llm_inference"] = llm_result
                result_data["symbol_inference_result"] = {
                    "inferred_name": llm_result["inferred_name"],
                    "ground_truth": func_name,
                    "is_correct": llm_result["inferred_name"].lower() == func_name.lower()
                }
                
                # Update token statistics
                if "token_stats" in llm_result:
                    token_stats_list.append(llm_result["token_stats"])
                
                # Update time statistics
                if "time_stats" in llm_result:
                    time_stats = llm_result["time_stats"]
                    elapsed = time_stats["elapsed_time"]
                    time_stats_list.append(time_stats)
                    
                    # Performance monitoring
                    if elapsed > 60:  # More than 1 minute
                        slow_inference_count += 1
                        logging.warning(f"⚠️ Slow inference detected: {elapsed:.2f}s (tokens: {llm_result.get('token_stats', {}).get('total_tokens', 'N/A')})")
                    else:
                        fast_inference_count += 1
                
                # Log inference result
                candidate_names = [f[0].func_name for f in similar_functions]
                logging.info(f"[LLM Inference] GT: {func_name} | Pred: {llm_result['inferred_name']} | Time: {elapsed:.2f}s | Candidates: {candidate_names}")
        
        all_results.append(result_data)
        
        # Save intermediate results and performance report every 1000 functions
        if (i + 1) % 1000 == 0:
            intermediate_file = save_intermediate_results(all_results, i + 1, total_functions)
            
            # Intermediate performance report
            if time_stats_list:
                recent_times = [stat["elapsed_time"] for stat in time_stats_list[-1000:]]
                avg_recent = sum(recent_times) / len(recent_times)
                max_recent = max(recent_times)
                min_recent = min(recent_times)
                
                logging.info(f"[Intermediate save] {i+1} functions processed")
                logging.info(f"Recent 1000 performance: avg {avg_recent:.2f}s, max {max_recent:.2f}s, min {min_recent:.2f}s")
                logging.info(f"Slow inferences ({slow_inference_count}) vs Fast inferences ({fast_inference_count})")
                logging.info(f"Result file: {intermediate_file}")
            
            # Memory cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save final statistics
    if token_stats_list and time_stats_list:
        save_statistics(token_stats_list, time_stats_list, slow_inference_count, fast_inference_count)

    # Process and save final results
    if all_results:
        logging.info(f"\nSymbol inference completed for {len(all_results)} functions.")
        
        # Count correct predictions
        correct_count = sum(1 for r in all_results 
                          if "symbol_inference_result" in r and r["symbol_inference_result"]["is_correct"])
        accuracy = correct_count / len(all_results) * 100 if all_results else 0
        
        # Prepare final results for output
        final_results = []
        for result in all_results:
            if "symbol_inference_result" in result:
                final_results.append({
                    "ground_truth": result["symbol_inference_result"]["ground_truth"],
                    "symbol": result["symbol_inference_result"]["inferred_name"]
                })
        
        # Save final output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_file = f"output_{timestamp}_llama_k-{k}.json"
        
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"\nFinal results saved to {final_output_file}")
        logging.info(f"Symbol inference results for {len(final_results)} functions saved.")
        logging.info(f"Accuracy: {correct_count}/{len(all_results)} ({accuracy:.1f}%)")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Ollama Symbol Inference System")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the kTrans model checkpoint")
    parser.add_argument("--db_path", type=str, required=True,
                       help="Path to the function database")
    parser.add_argument("--c_dir", type=str, required=True,
                       help="Directory containing C files")
    parser.add_argument("--pickle_dir", type=str, required=True,
                       help="Directory containing pickle files")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                       help="Device to use for inference (cuda/cpu)")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of similar functions to retrieve")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Batch size for embedding generation")
    parser.add_argument("--use_llm", action="store_true", default=True,
                       help="Enable LLM-based symbol inference")
    parser.add_argument("--ollama_model", type=str, default=DEFAULT_OLLAMA_MODEL,
                       help="Ollama model name to use")
    parser.add_argument("--resume_file", type=str, default=None,
                       help="Path to intermediate results file to resume from")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Path to log file")
    
    return parser.parse_args()

def main_with_args() -> None:
    """Main function with command line argument parsing"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Set global variables
    global RESUME_FILE, DEFAULT_OLLAMA_MODEL
    RESUME_FILE = args.resume_file
    DEFAULT_OLLAMA_MODEL = args.ollama_model
    
    # Validate paths
    if not os.path.exists(args.db_path):
        logging.error(f"Error: Database path does not exist: {args.db_path}")
        return
    
    if not os.path.exists(args.model_path):
        logging.error(f"Error: Model path does not exist: {args.model_path}")
        return
    
    if not os.path.exists(args.c_dir):
        logging.error(f"Error: C directory does not exist: {args.c_dir}")
        return
    
    # Collect all test set functions
    all_functions = collect_testset_functions([args.c_dir])
    logging.info(f"Total {len(all_functions)} function information collected.")
    
    if not all_functions:
        logging.warning("No functions to process. Exiting.")
        return
    
    # Load kTrans model
    logging.info('Loading kTrans model...')
    pl_model = AsmLMModule.load_from_checkpoint(args.model_path, total_steps=0)
    model = pl_model.bert
    model.to(torch.device(args.device))
    model.eval()
    
    # Load database
    db = FunctionDatabase(args.db_path)
    db.load()
    
    model_device = next(model.parameters()).device
    
    # Initialize Ollama model
    try:
        llm_model = initialize_ollama_client(model_name=args.ollama_model)
        if llm_model is None:
            logging.error("Ollama model initialization failed")
            return
        logging.info("Ollama model successfully initialized.")
    except Exception as e:
        logging.error(f"Error during Ollama model initialization: {e}")
        return

    # Resume functionality
    all_results = []
    resume_from = 0
    
    if args.resume_file and os.path.exists(args.resume_file):
        with open(args.resume_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        resume_from = len(all_results)
        logging.info(f"[Resume] Restored {resume_from} results from {args.resume_file}, continuing.")
    else:
        logging.info("[Resume] Starting from beginning.")

    total_functions = len(all_functions)
    
    # Statistics tracking
    token_stats_list = []
    time_stats_list = []
    slow_inference_count = 0
    fast_inference_count = 0
    
    # Process each function
    for i, func in enumerate(tqdm(all_functions[resume_from:], desc="Processing functions", 
                                 initial=resume_from, total=total_functions)):
        
        # Extract function information
        func_name = func["func_name"]
        func_addr = func["func_addr"]
        asm_code = func["asm_code"]
        decomp_code = func["decomp_code"]
        binary_name = func["binary"]
        
        # Generate embedding and find similar functions
        embedding = generate_embedding(model, asm_code, model_device)
        similar_functions = db.search_similar_functions(asm_code, embedding, k=args.k)
        
        # Prepare result data
        result_data = {
            "binary": binary_name,
            "function": {
                "name": func_name,
                "address": func_addr,
                "decomp_code": decomp_code
            },
            "similar_functions": []
        }
        
        # Add similar functions to result
        for j, (func_data, score) in enumerate(similar_functions, 1):
            result_data["similar_functions"].append({
                "name": func_data.func_name,
                "similarity": float(score),
                "asm_code": func_data.asm_code,
                "decomp_code": func_data.decomp_code
            })
        
        # Perform LLM inference if enabled
        if args.use_llm:
            debug_mode = (i == 0)
            llm_result = infer_symbol_with_llm(decomp_code, similar_functions, llm_model, debug=debug_mode)
            
            if llm_result:
                result_data["llm_inference"] = llm_result
                result_data["symbol_inference_result"] = {
                    "inferred_name": llm_result["inferred_name"],
                    "ground_truth": func_name,
                    "is_correct": llm_result["inferred_name"].lower() == func_name.lower()
                }
                
                # Update statistics
                if "token_stats" in llm_result:
                    token_stats_list.append(llm_result["token_stats"])
                if "time_stats" in llm_result:
                    time_stats = llm_result["time_stats"]
                    elapsed = time_stats["elapsed_time"]
                    time_stats_list.append(time_stats)
                    
                    if elapsed > 60:
                        slow_inference_count += 1
                    else:
                        fast_inference_count += 1
                
                # Log result
                candidate_names = [f[0].func_name for f in similar_functions]
                logging.info(f"[LLM] GT: {func_name} | Pred: {llm_result['inferred_name']} | Time: {elapsed:.2f}s | Candidates: {candidate_names}")
        
        all_results.append(result_data)
        
        # Save intermediate results every 1000 functions
        if (i + 1) % 1000 == 0:
            save_intermediate_results(all_results, i + 1, total_functions)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save final statistics and results
    if token_stats_list and time_stats_list:
        save_statistics(token_stats_list, time_stats_list, slow_inference_count, fast_inference_count)

    if all_results:
        logging.info(f"\nSymbol inference completed for {len(all_results)} functions.")
        
        # Prepare and save final results
        final_results = []
        correct_count = 0
        
        for result in all_results:
            if "symbol_inference_result" in result:
                final_results.append({
                    "ground_truth": result["symbol_inference_result"]["ground_truth"],
                    "symbol": result["symbol_inference_result"]["inferred_name"]
                })
                if result["symbol_inference_result"]["is_correct"]:
                    correct_count += 1
        
        # Save final output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_file = f"output_{timestamp}_llama_k-{args.k}.json"
        
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Final results saved to {final_output_file}")
        logging.info(f"Processed {len(final_results)} functions.")
        
        if final_results:
            accuracy = correct_count / len(final_results) * 100
            logging.info(f"Overall accuracy: {correct_count}/{len(final_results)} ({accuracy:.2f}%)")

if __name__ == "__main__":
    # Use command line arguments if available, otherwise use hardcoded main()
    import sys
    if len(sys.argv) > 1:
        main_with_args()
    else:
        main()