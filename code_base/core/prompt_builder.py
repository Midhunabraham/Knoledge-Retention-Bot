from typing import List, Tuple, Dict
from core.config import MAX_TOTAL_CONTEXT_CHARS
from core.llm_ollama import summarize_chunk

def build_prompt(question: str, hits: List[Tuple[str, Dict, float, str]]) -> Tuple[str, str]:
    sources_block = []
    total_chars = 0
    
    for i, (doc, meta, dist, _id) in enumerate(hits, start=1):
        title = meta.get("source", "Uploaded Doc") if isinstance(meta, dict) else "Uploaded Doc"
        
        # Summarize logic is now imported from llm_ollama
        chunk_text = summarize_chunk(doc)
        
        if total_chars + len(chunk_text) > MAX_TOTAL_CONTEXT_CHARS:
            remaining = MAX_TOTAL_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break
            chunk_text = chunk_text[:remaining]
        
        sources_block.append(f"[S{i}] {title}\n{chunk_text}")
        total_chars += len(chunk_text)
        if total_chars >= MAX_TOTAL_CONTEXT_CHARS:
            break

    context = "\n\n".join(sources_block) if sources_block else "No context available."
    
    system = (
        "You are an internal company knowledge assistant. Answer clearly and concisely. "
        "Cite sources as [S1], [S2]. If the answer is not in the provided context, say you don't know."
    )
    user = f"Question: {question}\n\nContext:\n{context}"
    return system, user