import os
import traceback
import streamlit as st
from typing import List
from openai import OpenAI
from chromadb.utils import embedding_functions
from core.config import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_embedder_openai():
    def openai_embed(texts: List[str]) -> List[List[float]]:
        embeddings = []
        resp = client.embeddings.create(model=DEFAULT_EMBED_MODEL, input=texts)
        for item in resp.data:
            embeddings.append(item.embedding)
        return embeddings

    class _Embedder(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return openai_embed(texts)

    return _Embedder()


def call_openai_chat(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model=DEFAULT_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❗ OpenAI API error:\n{e}\n\n{traceback.format_exc()}"
