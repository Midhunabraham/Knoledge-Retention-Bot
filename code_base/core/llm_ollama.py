import requests
import traceback
from core.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, SUMMARIZE_THRESHOLD, MAX_CHARS_PER_CHUNK

def call_ollama(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Calls local Ollama REST API. Tries /api/chat first, falls back to /api/generate."""
    base = OLLAMA_URL
    payload_chat = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(f"{base}/chat", json=payload_chat, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        # Fallback for older versions
        if isinstance(data, dict): return str(data)
        return resp.text
    except Exception as e_chat:
        # Fallback to /api/generate
        try:
            payload_gen = {"model": model, "prompt": f"{system_prompt}\n\n{user_prompt}", "stream": False}
            resp = requests.post(f"{base}/generate", json=payload_gen, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if "response" in data: return data["response"] # Newer Ollama
            if "text" in data: return data["text"]
            return resp.text
        except Exception as e_gen:
            tb = traceback.format_exc()
            return f"❗ Unable to call Ollama (chat error: {e_chat}; generate error: {e_gen}).\nTraceback:\n{tb}"

def summarize_chunk(chunk: str) -> str:
    """Summarize the chunk using Ollama if it's too long."""
    if len(chunk) <= SUMMARIZE_THRESHOLD:
        return chunk
    system_prompt = "You are a helpful assistant. Summarize the following text in concise key points."
    user_prompt = f"Text:\n{chunk}"
    summary = call_ollama(system_prompt, user_prompt)
    return summary[:MAX_CHARS_PER_CHUNK]