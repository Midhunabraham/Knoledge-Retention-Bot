import os
import streamlit as st
from groq import Groq
from core.config import GROQ_MODEL, SUMMARIZE_THRESHOLD, MAX_CHARS_PER_CHUNK

# Try to get API Key from Streamlit secrets or Environment Variable
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

def get_groq_client(api_key=None):
    key = api_key or GROQ_API_KEY
    if not key:
        return None
    return Groq(api_key=key)

def call_groq(system_prompt: str, user_prompt: str, model: str = GROQ_MODEL, api_key=None) -> str:
    """
    Calls Groq API (Llama 3, Mixtral, etc.)
    """
    client = get_groq_client(api_key)
    if not client:
        return "⚠️ Error: Groq API Key is missing."

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0.3, # Low temperature for factual RAG answers
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❗ Error calling Groq: {str(e)}"

def summarize_chunk(chunk: str, api_key=None) -> str:
    """
    Summarize the chunk using Groq (Llama 3.1 8b is great for this)
    """
    if len(chunk) <= SUMMARIZE_THRESHOLD:
        return chunk
    
    client = get_groq_client(api_key)
    if not client:
        # Fallback if no key
        return chunk[:MAX_CHARS_PER_CHUNK]

    prompt = f"Summarize this text into concise key points (max 3 sentences):\n\n{chunk}"
    
    try:
        # Use the "instant" model for summarization to keep it blazing fast
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant" 
        )
        return completion.choices[0].message.content
    except Exception:
        return chunk[:MAX_CHARS_PER_CHUNK]