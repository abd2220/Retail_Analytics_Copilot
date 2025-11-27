import os
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi

class LocalRetriever:
    def __init__(self, docs_path: str = "docs/"):
        self.docs_path = docs_path
        self.chunks: List[Dict] = []
        self.bm25 = None
        self._build_index()

    def _build_index(self):
        """Reads all .md files, chunks them, and builds BM25 index."""
        for filename in os.listdir(self.docs_path):
            if filename.endswith(".md"):
                filepath = os.path.join(self.docs_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    self._chunk_file(filename, content)
        
        corpus_tokens = [chunk["text"].lower().split() for chunk in self.chunks]
        if corpus_tokens:
            self.bm25 = BM25Okapi(corpus_tokens)

    def _chunk_file(self, filename: str, content: str):
        """Splits file into chunks, robustly handling different formats."""
        current_chunk_idx = 0
        
        # Split by markdown headers (##)
        sections = re.split(r'(^|\n)## ', content)
        
        # If the file has ## sections, process them
        if len(sections) > 1:
            main_title = sections[0].strip().replace("# ", "")
            for section in sections[1:]:
                if not section.strip(): continue
                lines = section.split('\n')
                section_title = lines[0].strip()
                body = '\n'.join(lines[1:]).strip()
                if not body: continue
                
                full_text = f"Source: {filename}\nContext: {main_title} > {section_title}\nContent:\n{body}"
                chunk_id = f"{filename}::chunk{current_chunk_idx}"
                self.chunks.append({"id": chunk_id, "text": full_text, "source": filename})
                current_chunk_idx += 1
        # --- ROBUST FALLBACK ---
        # If no ## sections, treat each list item as a chunk
        else:
            lines = content.split('\n')
            main_title = lines[0].strip().replace("# ", "")
            for line in lines[1:]:
                # If a line is a list item, it's a good chunk
                if line.strip().startswith('-'):
                    full_text = f"Source: {filename}\nContext: {main_title}\nContent:\n{line.strip()}"
                    chunk_id = f"{filename}::chunk{current_chunk_idx}"
                    self.chunks.append({"id": chunk_id, "text": full_text, "source": filename})
                    current_chunk_idx += 1

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Returns top_k chunks with scores."""
        if not self.bm25: return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        ranked_chunks = []
        for i, score in enumerate(scores):
            if score > 0.0: 
                chunk_data = self.chunks[i].copy()
                chunk_data["score"] = float(score)
                ranked_chunks.append(chunk_data)
        
        ranked_chunks.sort(key=lambda x: x["score"], reverse=True)
        return ranked_chunks[:top_k]

# --- Quick Test ---
if __name__ == "__main__":
    retriever = LocalRetriever()
    print(f"Loaded {len(retriever.chunks)} chunks.")
    
    test_q = "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer."
    print(f"\n--- Testing for '{test_q}' ---")
    results = retriever.search(test_q)
    for r in results:
        print(f"\nScore: {r['score']:.2f} | ID: {r['id']}")
        print(r['text'])