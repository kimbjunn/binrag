import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore

@dataclass
class FunctionData:
    func_id: str
    func_name: str
    asm_code: str
    decomp_code: str
    embedding_asm: np.ndarray = None

class FunctionDatabase:
    def __init__(self, save_dir: str = "function_db"):
        self.save_dir = save_dir
        self.asm_db = None
        self.function_map = {}
        self.next_func_id = 0
        
        # Create directory
        os.makedirs(save_dir, exist_ok=True)
        
    def add_function(self, func_name: str, asm_code: str, decomp_code: str, 
                    embedding_asm: np.ndarray) -> str:
        """Add new function data to DB"""
        func_id = f"func_{self.next_func_id}"
        self.next_func_id += 1
        
        func_data = FunctionData(
            func_id=func_id,
            func_name=func_name,
            asm_code=asm_code,
            decomp_code=decomp_code,
            embedding_asm=embedding_asm
        )
        
        self.function_map[func_id] = func_data
        return func_id
    
    def build_faiss_index(self):
        """Build FAISS index"""
        # Assembly code based index
        asm_embeddings = []
        asm_docs = []
        
        for func_id, func_data in self.function_map.items():
            if func_data.embedding_asm is not None:
                # Normalize embedding vector (L2 norm)
                embedding = func_data.embedding_asm / np.linalg.norm(func_data.embedding_asm)
                asm_embeddings.append(embedding)
                
                # Convert asm_code to string if it's a list
                asm_code_str = func_data.asm_code
                if isinstance(asm_code_str, list):
                    asm_code_str = "\n".join([str(item) for item in asm_code_str])
                
                asm_docs.append(Document(
                    page_content=asm_code_str,
                    metadata={"func_id": func_id, "func_name": func_data.func_name}
                ))
        
        # Create FAISS index
        if asm_embeddings:
            asm_embeddings = np.array(asm_embeddings, dtype=np.float32)
            dim = asm_embeddings.shape[1]
            
            # Create inner product based index (inner product of normalized vectors equals cosine similarity)
            asm_index = faiss.IndexFlatIP(dim)
            asm_index.add(asm_embeddings)
            
            self.asm_db = FAISS(
                index=asm_index,
                docstore=InMemoryDocstore(asm_docs),
                index_to_docstore_id={i: str(i) for i in range(len(asm_docs))},
                embedding_function=None  # Embeddings are already generated
            )
            
            print(f"FAISS index creation completed: {len(asm_embeddings)} vectors, {dim} dimensions")
    
    def search_similar_functions(self, asm_code, embedding_asm, k=5):
        """Search for similar functions"""
        if not self.asm_db:
            print("No FAISS index found. Please load the database first.")
            return []
        
        # Convert embedding vector to 2D array and convert to float32
        embedding_asm = np.array(embedding_asm).reshape(1, -1).astype(np.float32)
        
        # Normalize vector (L2 norm)
        embedding_asm = embedding_asm / np.linalg.norm(embedding_asm, axis=1, keepdims=True)
        
        # Print debug information
        print(f"\nSearch information:")
        print(f"Function map size: {len(self.function_map)}")
        print(f"FAISS index exists: {self.asm_db is not None}")
        print(f"Embedding dimensions: {embedding_asm.shape}")
        
        try:
            # Search directly in FAISS index
            D, I = self.asm_db.index.search(embedding_asm, k)
            
            print(f"\nSearch results:")
            print(f"Number of search results: {len(I[0])}")
            print(f"Distances (D): {D[0]}")
            print(f"Indices (I): {I[0]}")
            
            # Process results
            similar_functions = []
            for score, idx in zip(D[0], I[0]):
                # Check if index is within valid range
                if idx < len(self.function_map):
                    # Get function ID from function ID list at the corresponding index
                    func_ids = list(self.function_map.keys())
                    if idx < len(func_ids):
                        func_id = func_ids[idx]
                        similar_functions.append((self.function_map[func_id], float(score)))
                        print(f"Function: {self.function_map[func_id].func_name}, Score: {float(score):.4f}")
                    else:
                        print(f"Warning: Function ID not found for index {idx}.")
                else:
                    print(f"Warning: Index {idx} is invalid.")
            
            # Select randomly if no similar functions found
            if not similar_functions and self.function_map:
                print("No similar functions found. Selecting functions randomly.")
                import random
                random_funcs = random.sample(list(self.function_map.values()), min(k, len(self.function_map)))
                similar_functions = [(func, 0.0) for func in random_funcs]  # Set similarity score to 0.0
            
            return similar_functions
            
        except Exception as e:
            print(f"Error occurred during search: {str(e)}")
            # Select functions randomly when error occurs
            if self.function_map:
                print("Error occurred. Selecting functions randomly.")
                import random
                random_funcs = random.sample(list(self.function_map.values()), min(k, len(self.function_map)))
                return [(func, 0.0) for func in random_funcs]
            return []
    
    def save(self):
        """Save DB"""
        # Save function map
        with open(os.path.join(self.save_dir, "semantic_db.pkl"), "wb") as f:
            pickle.dump(self.function_map, f)
        
        # Save FAISS index
        if self.asm_db is not None:
            self.asm_db.save_local(os.path.join(self.save_dir, "asm_db"))
    
    def load(self):
        """Load DB"""
        # Load function map
        map_path = os.path.join(self.save_dir, "semantic_db.pkl")
        if os.path.exists(map_path):
            with open(map_path, "rb") as f:
                self.function_map = pickle.load(f)
        
        # Load FAISS index
        asm_db_path = os.path.join(self.save_dir, "asm_db")
        if os.path.exists(asm_db_path):
            # Load with embedding function set to None
            # Actual embeddings are already stored in the index, so not needed
            # Set allow_dangerous_deserialization=True to ignore security warnings
            self.asm_db = FAISS.load_local(asm_db_path, embeddings=None, allow_dangerous_deserialization=True)