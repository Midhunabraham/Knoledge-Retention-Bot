import requests

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "mistral",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "stream": False
    }
)

print(resp.status_code, resp.json())
