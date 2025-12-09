import os
import sys
import requests
from dotenv import load_dotenv

# Add src to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.utils.keychain_manager import get_api_key
except ImportError:
    print("⚠️  Could not import keychain_manager. Using .env only.")
    get_api_key = lambda x: None

# Load environment variables from .env file
load_dotenv()

# Try to get the key from common variable names in .env
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# If not in .env, try Keychain
if not api_key:
    print("ℹ️  Checking macOS Keychain...")
    api_key = get_api_key("GOOGLE_API_KEY") or get_api_key("GEMINI_API_KEY")

if not api_key:
    print(
        "❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env file, environment, or macOS Keychain."
    )
    print("👉 Please add it to .env or run scripts/setup/store_api_keys.sh")
    exit(1)

print(f"ℹ️  Found API Key: {api_key[:5]}...{api_key[-4:]}")

# URL for Gemini Pro (using v1beta API)
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}
data = {
    "contents": [
        {"parts": [{"text": "Reply with 'Yes, I am working!' if you receive this."}]}
    ]
}

try:
    print("Testing API key...")
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("✅ Success! The Gemini API key is working.")
        try:
            print(
                "Response:",
                response.json()["candidates"][0]["content"]["parts"][0]["text"],
            )
        except:
            print("Response received (raw):", response.text)
    else:
        print(f"❌ Failed. Status Code: {response.status_code}")
        print("Error:", response.text)

except Exception as e:
    print(f"❌ Exception occurred: {e}")
