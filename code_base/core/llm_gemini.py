import google.generativeai as genai
import streamlit as st
import os
from core.config import SUMMARIZE_THRESHOLD, MAX_CHARS_PER_CHUNK

# Try to get API Key from Streamlit secrets or Environment Variable
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

def configure_gemini(api_key=None):
    """
    Configures the Gemini API. 
    Can be called from the frontend if the key is provided via text input.
    """
    key_to_use = api_key or GEMINI_API_KEY
    if not key_to_use:
        raise ValueError("Gemini API Key is missing.")
    genai.configure(api_key=key_to_use)

def call_gemini(system_prompt: str, user_prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    """
    Calls Google Gemini API.
    """
    try:
        model = genai.GenerativeModel(model_name)
        
        # Gemini doesn't have a strict "system" role in the same way as OpenAI/Ollama in the simple API,
        # but we can prepend it to the prompt or use system_instruction if using the beta API.
        # For stability, we merge them here.
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❗ Error calling Gemini: {str(e)}"

def summarize_chunk(chunk: str, api_key=None) -> str:
    """
    Summarize the chunk using Gemini if it's too long.
    """
    if len(chunk) <= SUMMARIZE_THRESHOLD:
        return chunk
        
    # Ensure configured
    if api_key: 
        configure_gemini(api_key)
    elif not GEMINI_API_KEY:
        # If no key is available, return truncated text to avoid crashing
        return chunk[:MAX_CHARS_PER_CHUNK]

    prompt = f"Summarize the following text in concise key points:\n\n{chunk}"
    
    try:
        # Use Flash for faster/cheaper summarization
        model = genai.GenerativeModel("gemini-1.5-flash") 
        response = model.generate_content(prompt)
        return response.text[:MAX_CHARS_PER_CHUNK]
    except Exception:
        return chunk[:MAX_CHARS_PER_CHUNK]