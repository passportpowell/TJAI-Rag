import pandas as pd
import numpy as np
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import time
import re
from typing import List, Dict, Tuple

class LocalRAGSystem:
    def __init__(self, csv_path: str, index_dir: str = None):
        """
        Initialize the RAG system with the CSV file path and optional index directory.
        
        Args:
            csv_path: Path to the CSV file
            index_dir: Directory to save/load the FAISS index and metadata
        """
        self.csv_path = csv_path
        self.index_dir = index_dir or os.path.join(os.path.dirname(csv_path), "rag_index")
        self.model = None
        self.df = None
        self.index = None
        self.id_to_row_map = {}

        # Create index directory if it doesn't exist
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
    
    def load_data(self) -> None:
        """Load the CSV data into a pandas DataFrame."""
        start_time = time.time()
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            print(f"Column names: {', '.join(self.df.columns)}")
            print(f"Data loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise
    
    def load_model(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Load the sentence transformer model for creating embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        start_time = time.time()
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _prepare_text_for_embedding(self, row) -> str:
        """
        Prepare text from a DataFrame row for embedding by combining relevant fields.
        
        Args:
            row: A pandas Series representing a row in the DataFrame
            
        Returns:
            A combined string suitable for embedding
        """
        # Combine relevant fields, focusing on those likely to contain meaningful content
        fields = []
        
        if 'objective_title' in row and pd.notna(row['objective_title']):
            fields.append(f"Objective: {row['objective_title']}")
            
        if 'task_title' in row and pd.notna(row['task_title']):
            fields.append(f"Task: {row['task_title']}")
            
        if 'sub_task_title' in row and pd.notna(row['sub_task_title']):
            fields.append(f"Sub-task: {row['sub_task_title']}")
            
        if 'country' in row and pd.notna(row['country']):
            fields.append(f"Country: {row['country']}")
            
        if 'final_copy' in row and pd.notna(row['final_copy']):
            fields.append(f"Final Copy: {row['final_copy']}")
            
        if 'summary' in row and pd.notna(row['summary']):
            fields.append(f"Summary: {row['summary']}")
        
        # Join all fields with separator
        return " | ".join(fields)
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build the FAISS index from the CSV data.
        
        Args:
            force_rebuild: If True, rebuild even if index files exist
        """
        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        metadata_path = os.path.join(self.index_dir, "metadata.pkl")
        
        # Check if index already exists and we're not forcing a rebuild
        if os.path.exists(index_path) and os.path.exists(metadata_path) and not force_rebuild:
            print("Loading existing index...")
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.id_to_row_map = pickle.load(f)
            print(f"Loaded existing index with {self.index.ntotal} vectors")
            return
        
        if self.df is None:
            self.load_data()
            
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        print("Building index...")
        
        # Prepare text for each row
        texts = []
        for idx, row in self.df.iterrows():
            text = self._prepare_text_for_embedding(row)
            texts.append(text)
            self.id_to_row_map[idx] = idx  # Map FAISS ID to DataFrame index
        
        # Create embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
        self.index.add(embeddings)
        
        # Save index and metadata
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.id_to_row_map, f)
        
        print(f"Index built with {len(texts)} entries in {time.time() - start_time:.2f} seconds")
        print(f"Index and metadata saved to {self.index_dir}")
    
    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        Query the RAG system with a text question.
        
        Args:
            query_text: The query text
            n_results: Number of top results to return
            
        Returns:
            List of dictionaries containing the matching rows and their scores
        """
        if self.index is None:
            self.build_index()
            
        if self.model is None:
            self.load_model()
        
        # Create query embedding
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, n_results)
        
        # Get the actual data for each result
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for padded results when not enough matches found
                continue
            
            # Get the original row index from the mapping
            df_idx = self.id_to_row_map[idx]
            row_data = self.df.iloc[df_idx].to_dict()
            
            # Add the similarity score
            result = {
                "rank": i + 1,
                "score": float(score),
                "data": row_data
            }
            results.append(result)
        
        return results
    
    def format_results(self, results: List[Dict], detailed: bool = False) -> str:
        """
        Format the query results into a readable string.
        
        Args:
            results: List of result dictionaries
            detailed: Whether to include all fields in the output
            
        Returns:
            Formatted string of results
        """
        if not results:
            return "No results found."
        
        formatted = []
        for result in results:
            rank = result["rank"]
            score = result["score"]
            data = result["data"]
            
            entry = [f"Result #{rank} (Score: {score:.4f})"]
            
            # Always include key fields
            if 'objective_title' in data and pd.notna(data['objective_title']):
                entry.append(f"Objective: {data['objective_title']}")
                
            if 'task_title' in data and pd.notna(data['task_title']):
                entry.append(f"Task: {data['task_title']}")
                
            if 'sub_task_title' in data and pd.notna(data['sub_task_title']):
                entry.append(f"Sub-task: {data['sub_task_title']}")
            
            # Add summary or final copy depending on detail level
            if detailed:
                # Include all fields in detailed view
                for key, value in data.items():
                    if key not in ['objective_title', 'task_title', 'sub_task_title'] and pd.notna(value):
                        entry.append(f"{key}: {value}")
            else:
                # Just include summary in brief view
                if 'summary' in data and pd.notna(data['summary']):
                    entry.append(f"Summary: {data['summary']}")
            
            formatted.append("\n".join(entry))
        
        return "\n\n".join(formatted)
    
    def interactive_query(self):
        """Start an interactive query session."""
        print("\n=== Joule Model Report RAG System ===")
        print("Type your query or 'exit' to quit")
        
        while True:
            query = input("\nEnter query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            if not query:
                continue
                
            detailed = False
            if query.lower().endswith(" -d") or query.lower().endswith(" --detailed"):
                detailed = True
                query = re.sub(r'\s+(-d|--detailed)$', '', query)
            
            start_time = time.time()
            try:
                results = self.query(query)
                formatted = self.format_results(results, detailed)
                print(f"\n{formatted}")
                print(f"\nQuery completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error processing query: {e}")


if __name__ == "__main__":
    # Set up the RAG system
    csv_path = r"D:\OneDrive\Python Scripts\Joule Model Report.csv"
    rag = LocalRAGSystem(csv_path)
    
    # Build or load the index
    print("Initializing RAG system...")
    rag.load_data()
    rag.build_index()
    
    # Start interactive query session
    rag.interactive_query()