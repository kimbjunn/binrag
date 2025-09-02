import os
import re
import pickle
import logging
import psutil
import time
import hashlib
import signal
import atexit
import gc
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Generator, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import partial, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
import faiss

from function_database import FunctionDatabase
from asmlm import AsmLMModule, TOKENIZER


@dataclass
class ProcessingConfig:
    """Configuration class for processing parameters"""
    batch_size: int = 32
    max_seq_len: int = 512
    use_amp: bool = True
    device: str = 'cuda'
    num_workers: int = None
    memory_limit_gb: float = 8.0
    prefetch_factor: int = 2
    pin_memory: bool = True
    enable_jit: bool = True


@dataclass
class ProcessingStats:
    """Statistics tracking for processing operations"""
    total_functions: int = 0
    processed_functions: int = 0
    skipped_functions: int = 0
    memory_peaks: List[float] = None
    processing_times: List[float] = None
    
    def __post_init__(self):
        if self.memory_peaks is None:
            self.memory_peaks = []
        if self.processing_times is None:
            self.processing_times = []


class LoggingManager:
    """Centralized logging management"""
    
    @staticmethod
    def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
        """Setup improved logging with file and console handlers"""
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s'
        )  
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger


class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, limit_gb: float = 8.0):
        self.limit_bytes = limit_gb * 1024 * 1024 * 1024
        self.peak_usage = 0
        
    def check_memory(self) -> Tuple[float, bool]:
        """Check current memory usage and limit status"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        usage_gb = mem_info.rss / 1024 / 1024 / 1024
        
        self.peak_usage = max(self.peak_usage, usage_gb)
        is_over_limit = mem_info.rss > self.limit_bytes
        
        return usage_gb, is_over_limit
    
    def log_gpu_memory(self, logger: logging.Logger):
        """Log GPU memory usage for all available devices"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                logger.info(f"GPU {i} memory: {allocated:.2f}MB / {reserved:.2f}MB")


class FileProcessor:
    """Handle file processing operations with optimization"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_file_hash(file_path: str) -> str:
        """Calculate and cache file hash"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    @staticmethod
    def parse_c_functions(c_file_path: str, logger: logging.Logger) -> List[Tuple[str, str]]:
        """Parse C functions from file with memory optimization"""
        try:
            file_size = os.path.getsize(c_file_path)
            
            with open(c_file_path, 'r', encoding='utf-8') as f:
                if file_size > 100 * 1024 * 1024:  # 100MB+
                    return FileProcessor._parse_large_file(f)
                else:
                    content = f.read()
                    
            pattern = re.compile(
                r'/\* Function: (.+?) at (0x[0-9A-Fa-f]+) \*/\s*([\s\S]+?\{[\s\S]*?\})',
                re.MULTILINE
            )
            
            functions = []
            for match in pattern.finditer(content):
                func_name = match.group(1).strip()
                func_addr = match.group(2).strip()
                func_decl_and_body = match.group(3).strip()
                full_func = f"{func_name}@{func_addr}\n{func_decl_and_body}"
                functions.append((f"{func_name}@{func_addr}", full_func))
                
            return functions
            
        except Exception as e:
            logger.error(f"File parsing error {c_file_path}: {e}")
            return []
    
    @staticmethod
    def _parse_large_file(file_obj) -> List[Tuple[str, str]]:
        """Parse large files in chunks to manage memory"""
        functions = []
        chunk_size = 1024 * 1024  # 1MB chunks
        buffer = ""
        
        pattern = re.compile(
            r'/\* Function: (.+?) at (0x[0-9A-Fa-f]+) \*/\s*([\s\S]+?\{[\s\S]*?\})',
            re.MULTILINE
        )
        
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
                
            buffer += chunk
            matches = list(pattern.finditer(buffer))
            
            if matches:
                last_match = matches[-1]
                buffer = buffer[last_match.end():]
                
                for match in matches[:-1]:
                    func_name = match.group(1).strip()
                    func_addr = match.group(2).strip()
                    func_decl_and_body = match.group(3).strip()
                    full_func = f"{func_name}@{func_addr}\n{func_decl_and_body}"
                    functions.append((f"{func_name}@{func_addr}", full_func))
        
        return functions
    
    @staticmethod
    def load_assembly_code(pickle_path: str, logger: logging.Logger) -> Dict[str, List[Tuple]]:
        """Load and process assembly code from pickle file"""
        try:
            file_size = os.path.getsize(pickle_path)
            if file_size > 500 * 1024 * 1024:  # 500MB+
                logger.warning(f"Large pickle file: {file_size / 1024 / 1024:.2f}MB")
            
            with open(pickle_path, 'rb') as f:
                func_list = pickle.load(f)
            
            asm_functions = {}
            duplicate_count = 0
            
            for func in func_list:
                if not func:
                    continue
                    
                raw_addr = func[0][0]
                addr_int = int(raw_addr, 16) if isinstance(raw_addr, str) else int(raw_addr)
                func_addr = hex(addr_int)
                
                if func_addr in asm_functions:
                    duplicate_count += 1
                    continue
                asm_functions[func_addr] = func
            
            if duplicate_count:
                logger.info(f"Removed {duplicate_count} duplicate addresses from pickle")
                
            return asm_functions
            
        except Exception as e:
            logger.error(f"Assembly loading error: {pickle_path}, {e}")
            return {}


class ModelManager:
    """Handle model setup and optimization"""
    
    @staticmethod
    def setup_model(model_path: str, config: ProcessingConfig, logger: logging.Logger) -> Tuple[torch.nn.Module, str]:
        """Setup and optimize model for processing"""
        device = config.device
        
        if not torch.cuda.is_available() and device == 'cuda':
            logger.warning("CUDA not available, switching to CPU")
            device = 'cpu'
            config.device = 'cpu'
        
        # Load model
        pl = AsmLMModule.load_from_checkpoint(model_path, total_steps=0)
        model = pl.bert
        model.eval()
        
        # Apply JIT compilation if enabled
        if device == 'cuda' and config.enable_jit:
            model = ModelManager._apply_jit_compilation(model, device, logger)
        elif device == 'cuda' and not config.enable_jit:
            logger.info("JIT compilation disabled")
        
        model.to(device)
        
        # Setup multi-GPU if available
        if device == 'cuda' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"DataParallel setup: {torch.cuda.device_count()} GPUs")
        
        return model, device
    
    @staticmethod
    def _apply_jit_compilation(model: torch.nn.Module, device: str, logger: logging.Logger) -> torch.nn.Module:
        """Apply JIT compilation with warning suppression"""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                
                dummy_inputs = tuple(
                    torch.zeros(2, 64, dtype=torch.long).to(device) 
                    for _ in range(7)
                )
                
                model = torch.jit.trace(model, dummy_inputs)
                logger.info("Model JIT compilation completed (TracerWarning suppressed)")
                
        except Exception as e:
            logger.warning(f"JIT compilation failed, using regular model: {e}")
            # Reload original model on failure
            pl = AsmLMModule.load_from_checkpoint("", total_steps=0)  # This would need the path
            model = pl.bert
            model.eval()
            
        return model


class EmbeddingGenerator:
    """Handle embedding generation with optimization"""
    
    @staticmethod
    def generate_embeddings(
        model: torch.nn.Module,
        functions: List[Dict],
        config: ProcessingConfig,
        memory_monitor: MemoryMonitor,
        logger: logging.Logger
    ) -> Tuple[np.ndarray, List[Dict], ProcessingStats]:
        """Generate embeddings for all functions"""
        
        device = config.device
        stats = ProcessingStats()
        total_functions = len(functions)
        
        logger.info(f"Starting embedding generation: {total_functions} functions")
        
        all_embeddings = np.zeros((total_functions, 768))  # BERT embedding dimension
        current_batch_size = config.batch_size
        
        logger.info(f"Initial batch size: {current_batch_size}")
        
        with tqdm(total=total_functions, desc="Generating embeddings") as pbar:
            start_idx = 0
            batch_count = 0
            
            while start_idx < total_functions:
                batch_count += 1
                
                try:
                    # Progress logging
                    EmbeddingGenerator._log_progress(batch_count, logger)
                    
                    # Memory management
                    mem_usage = EmbeddingGenerator._handle_memory_check(memory_monitor, device, logger)
                    
                    # Process batch
                    end_idx = min(start_idx + current_batch_size, total_functions)
                    batch_functions = functions[start_idx:end_idx]
                    batch_size_actual = len(batch_functions)
                    
                    # Generate embeddings for batch
                    embeddings_batch, processing_time = EmbeddingGenerator._process_batch(
                        model, batch_functions, device, config
                    )
                    
                    # Store results
                    for i, embedding in enumerate(embeddings_batch):
                        all_embeddings[start_idx + i] = embedding
                    
                    # Update statistics
                    EmbeddingGenerator._update_stats(stats, batch_size_actual, processing_time, mem_usage)
                    
                    # Update progress bar
                    pbar.update(batch_size_actual)
                    pbar.set_postfix({
                        'mem': f'{mem_usage:.1f}GB',
                        'batch_time': f'{processing_time:.2f}s',
                        'batch_size': current_batch_size
                    })
                    
                    start_idx = end_idx
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e) and current_batch_size > 1:
                        current_batch_size = EmbeddingGenerator._handle_oom(e, current_batch_size, logger)
                    else:
                        logger.error(f"Error during embedding generation: {e}")
                        raise
        
        stats.total_functions = total_functions
        stats.skipped_functions = 0
        
        logger.info(f"Embedding completed: {total_functions} functions")
        return all_embeddings, functions, stats
    
    @staticmethod
    def _log_progress(batch_count: int, logger: logging.Logger):
        """Log processing progress"""
        if batch_count == 1:
            logger.info(f"Starting first batch processing (batch {batch_count})")
        elif batch_count % 10 == 0:
            logger.info(f"Processing batch {batch_count}...")
    
    @staticmethod
    def _handle_memory_check(memory_monitor: MemoryMonitor, device: str, logger: logging.Logger) -> float:
        """Check and handle memory usage"""
        mem_usage, is_over_limit = memory_monitor.check_memory()
        if is_over_limit:
            logger.warning(f"Memory limit exceeded: {mem_usage:.2f}GB")
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        return mem_usage
    
    @staticmethod
    def _process_batch(
        model: torch.nn.Module,
        batch_functions: List[Dict],
        device: str,
        config: ProcessingConfig
    ) -> Tuple[List[np.ndarray], float]:
        """Process a single batch and return embeddings"""
        start_time = time.time()
        
        # Prepare batch data
        encs = [TOKENIZER.encode_func(f['asm_code']) for f in batch_functions]
        batch_data = {
            k: torch.stack([torch.tensor(enc[k]) for enc in encs]).to(device)
            for k in [
                'func_token_ids', 'func_insn_type_ids', 'func_opnd_type_ids',
                'func_reg_id_ids', 'func_opnd_r_ids', 'func_opnd_w_ids', 'func_eflags_ids'
            ]
        }
        
        # Generate embeddings
        with autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=config.use_amp):
            with torch.no_grad():
                out = model(
                    batch_data['func_token_ids'],
                    batch_data['func_insn_type_ids'],
                    batch_data['func_opnd_type_ids'],
                    batch_data['func_reg_id_ids'],
                    batch_data['func_opnd_r_ids'],
                    batch_data['func_opnd_w_ids'],
                    batch_data['func_eflags_ids']
                )[:, 0, :]
        
        # Normalize embeddings
        norm = torch.norm(out, dim=1, keepdim=True)
        normalized_out = out / norm
        cpu_embeddings = normalized_out.cpu().numpy()
        
        # Cleanup
        del batch_data, out, normalized_out, encs
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        processing_time = time.time() - start_time
        return [cpu_embeddings[i] for i in range(len(batch_functions))], processing_time
    
    @staticmethod
    def _update_stats(stats: ProcessingStats, batch_size: int, processing_time: float, mem_usage: float):
        """Update processing statistics"""
        stats.processed_functions += batch_size
        stats.processing_times.append(processing_time)
        stats.memory_peaks.append(mem_usage)
    
    @staticmethod
    def _handle_oom(error: RuntimeError, current_batch_size: int, logger: logging.Logger) -> int:
        """Handle out of memory error by reducing batch size"""
        torch.cuda.empty_cache()
        new_batch_size = max(1, current_batch_size // 2)
        logger.warning(f"OOM occurred: batch size {current_batch_size} -> {new_batch_size}")
        return new_batch_size


class FunctionMatcher:
    """Handle function matching between C and assembly files"""
    
    @staticmethod
    def match_functions(root_dir: str, logger: logging.Logger) -> List[Dict]:
        """Match functions from C and pickle files"""
        c_files = {f[:-2]: f for f in os.listdir(root_dir) if f.endswith('.c')}
        pkl_files = {f[:-4]: f for f in os.listdir(root_dir) if f.endswith('.pkl')}
        common_files = list(set(c_files) & set(pkl_files))
        
        results = []
        max_workers = min(8, len(common_files))
        work_args = [(name, root_dir) for name in common_files]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(FunctionMatcher._process_file_pair, args): args[0] 
                for args in work_args
            }
            
            for future in tqdm(as_completed(future_to_name), total=len(future_to_name), desc="Matching functions"):
                try:
                    result = future.result(timeout=60)
                    if result:
                        results.extend(result)
                except Exception as e:
                    name = future_to_name[future]
                    logger.error(f"File processing error {name}: {e}")
        
        return results
    
    @staticmethod
    def _process_file_pair(args: Tuple[str, str]) -> List[Dict]:
        """Process a pair of C and pickle files"""
        name, root_dir = args
        c_path = os.path.join(root_dir, f"{name}.c")
        pkl_path = os.path.join(root_dir, f"{name}.pkl")
        
        try:
            # Use a temporary logger for this process
            temp_logger = logging.getLogger(f"process_{name}")
            
            funcs = FileProcessor.parse_c_functions(c_path, temp_logger)
            asm_map = FileProcessor.load_assembly_code(pkl_path, temp_logger)
        except Exception as e:
            return []
        
        results = []
        for sig, full_func in funcs:
            try:
                func_name, func_addr = sig.split('@')
                norm_addr = FunctionMatcher._normalize_hex_address(func_addr)
                
                if norm_addr not in asm_map:
                    continue
                    
                results.append({
                    "func_name": func_name,
                    "func_addr": norm_addr,
                    "decomp_code": full_func,
                    "asm_code": asm_map[norm_addr],
                    "binary": name,
                    "software": "root"
                })
            except ValueError:
                continue
        
        return results
    
    @staticmethod
    def _normalize_hex_address(addr: str) -> str:
        """Normalize hexadecimal address format"""
        addr = addr.strip().lower()
        if addr.startswith('0x'):
            return hex(int(addr, 16))
        return hex(int(addr, 16) if all(c in '0123456789abcdef' for c in addr) else int(addr))


class FaissIndexBuilder:
    """Handle FAISS index building and optimization"""
    
    @staticmethod
    def build_optimized_index(db: FunctionDatabase, device: str, logger: logging.Logger) -> None:
        """Build optimized FAISS index with GPU support"""
        try:
            # Build initial CPU index
            db.build_faiss_index()
            cpu_index = db.asm_db.index
            db._cpu_index = cpu_index
            
            # Attempt GPU optimization
            if device == 'cuda' and torch.cuda.is_available():
                FaissIndexBuilder._setup_gpu_index(db, cpu_index, logger)
            
        except Exception as e:
            logger.error(f"FAISS index building error: {e}")
    
    @staticmethod
    def _setup_gpu_index(db: FunctionDatabase, cpu_index, logger: logging.Logger):
        """Setup GPU-optimized FAISS index"""
        ngpu = torch.cuda.device_count()
        dim = cpu_index.d
        
        try:
            vectors = faiss.vector_to_array(cpu_index.codes).reshape(-1, dim).astype(np.float32)
            
            if vectors.shape[0] > 0:
                gpu_resources = [faiss.StandardGpuResources() for _ in range(ngpu)]
                
                # Optimize memory usage
                for res in gpu_resources:
                    res.setTempMemory(512 * 1024 * 1024)  # 512MB
                
                config = faiss.GpuIndexFlatConfig()
                config.useFloat16 = False
                
                if ngpu == 1:
                    # Single GPU setup
                    gpu_idx = faiss.GpuIndexFlatL2(gpu_resources[0], dim, config)
                    gpu_idx.add(vectors)
                    db.asm_db.index = gpu_idx
                else:
                    # Multi-GPU sharding
                    shard = faiss.IndexShards(dim, True, True)
                    for dev in range(ngpu):
                        config.device = dev
                        gpu_idx = faiss.GpuIndexFlatL2(gpu_resources[dev], dim, config)
                        gpu_idx.add(vectors[dev::ngpu])
                        shard.add_shard(gpu_idx)
                    db.asm_db.index = shard
                
                logger.info(f"GPU index creation completed: {ngpu} GPUs")
            
        except Exception as e:
            logger.error(f"GPU index creation failed: {e}")
            db.asm_db.index = cpu_index
            logger.info("Using CPU index")


class DatabaseBuilder:
    """Main database building orchestrator"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.logger = LoggingManager.setup_logging()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup cleanup signal handlers"""
        signal.signal(signal.SIGINT, self._cleanup_handler)
        signal.signal(signal.SIGTERM, self._cleanup_handler)
    
    def _cleanup_handler(self, signum, frame):
        """Handle cleanup on termination"""
        self.logger.info("Performing cleanup...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cleanup completed")
    
    def build_database(self, root_dir: str, model_path: str, save_dir: str) -> ProcessingStats:
        """Build optimized database with full pipeline"""
        self.logger.info("=== Starting advanced database building ===")
        start_time = time.time()
        
        # Initialize components
        memory_monitor = MemoryMonitor(self.config.memory_limit_gb)
        
        # Setup model
        model, device = ModelManager.setup_model(model_path, self.config, self.logger)
        memory_monitor.log_gpu_memory(self.logger)
        
        # Match functions
        self.logger.info("Starting function matching")
        functions = FunctionMatcher.match_functions(root_dir, self.logger)
        functions = [f for f in functions if f.get('asm_code')]
        
        if not functions:
            self.logger.warning("No functions to process.")
            return ProcessingStats()
        
        self.logger.info(f"Total {len(functions)} functions found")
        
        # Generate embeddings
        embeddings, filtered_functions, stats = EmbeddingGenerator.generate_embeddings(
            model, functions, self.config, memory_monitor, self.logger
        )
        
        # Clean up model memory
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Build database
        self.logger.info("Building FAISS database")
        db = FunctionDatabase(save_dir)
        
        for func, emb in tqdm(zip(filtered_functions, embeddings), desc="Adding functions"):
            db.add_function(
                func_name=func['func_name'],
                asm_code=func['asm_code'],
                decomp_code=func['decomp_code'],
                embedding_asm=emb
            )
        
        # Build and optimize index
        FaissIndexBuilder.build_optimized_index(db, device, self.logger)
        
        # Save database
        db.save()
        
        # Final statistics and reporting
        total_time = time.time() - start_time
        stats.total_functions = len(functions)
        
        self._log_final_stats(stats, total_time, memory_monitor)
        
        return stats
    
    def _log_final_stats(self, stats: ProcessingStats, total_time: float, memory_monitor: MemoryMonitor):
        """Log final processing statistics"""
        self.logger.info("=== Building completed ===")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Processed functions: {stats.processed_functions}")
        self.logger.info(f"Average processing time: {np.mean(stats.processing_times):.3f} seconds/batch")
        self.logger.info(f"Peak memory usage: {memory_monitor.peak_usage:.2f}GB")
        self.logger.info(f"Processing efficiency: {stats.processed_functions / total_time:.2f} functions/second")


def main():
    """Main entry point"""
    # Configuration
    config = ProcessingConfig(
        batch_size=32,
        max_seq_len=512,
        use_amp=True,
        device='cuda',
        num_workers=mp.cpu_count() - 1,
        memory_limit_gb=32.0,
        prefetch_factor=2,
        pin_memory=True,
        enable_jit=False  # Disable to prevent TracerWarning
    )
    
    # Paths
    root_dir = r"E:\symbol-infer\dataset\malware\MalwareSourceCode\dataset\division_malware\db_set"
    model_path = r"E:\symbol-infer\embedding\kTrans-release-main\final-code\pretrained\ktrans-110M-epoch-2.ckpt"
    save_dir = r"E:\symbol-infer\embedding\kTrans-release-main\final-code\db\test"
    
    # Build database
    builder = DatabaseBuilder(config)
    
    # Log GPU information
    if torch.cuda.is_available():
        builder.logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            builder.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.2f}GB)")
    
    # Execute database building
    start_time = time.time()
    stats = builder.build_database(root_dir, model_path, save_dir)
    total_time = time.time() - start_time
    
    builder.logger.info(f"Total completion time: {total_time:.2f} seconds")


if __name__ == '__main__':
    main()