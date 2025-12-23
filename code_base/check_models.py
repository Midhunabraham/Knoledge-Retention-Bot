import google.generativeai as genai
import os
import getpass

print("--- Gemini Model Checker ---")

# 1. Try to get key from environment
api_key = os.getenv("GEMINI_API_KEY")

# 2. If not found, ask the user to paste it
if not api_key:
    print("⚠️  Env variable 'GEMINI_API_KEY' not found.")
    api_key = getpass.getpass(prompt="🔑 Please paste your Google AI Studio Key here (hidden input): ")

if not api_key:
    print("❌ Error: You must provide an API Key.")
    exit()

# 3. Configure and Check
try:
    genai.configure(api_key=api_key)
    print("\n✅ Key accepted! Fetching available models...")
    
    found_any = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   • {m.name}")
            found_any = True
            
    if not found_any:
        print("\n⚠️  No chat models found. Your key might be invalid.")
    else:
        print("\n🚀 Success! Use one of the names above in your 'app_gemini.py'.")

except Exception as e:
    print(f"\n❌ connection failed: {e}")