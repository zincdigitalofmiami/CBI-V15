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
api_key = os.getenv("OPENAI_API_KEY")

# If not in .env, try Keychain
if not api_key:
    print("ℹ️  Checking macOS Keychain...")
    api_key = get_api_key("OPENAI_API_KEY")

if not api_key:
    print(
        "❌ Error: OPENAI_API_KEY not found in .env file, environment, or macOS Keychain."
    )
    print("👉 Please add it to .env or run scripts/setup/store_api_keys.sh")
    exit(1)

print(f"ℹ️  Found API Key: {api_key[:5]}...{api_key[-4:]}")

# URL for OpenAI Models API
url = "https://api.openai.com/v1/models"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

try:
    print("Testing API key...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("✅ Success! The OpenAI API key is working.")
        models = [m["id"] for m in response.json()["data"]]
        print(f"Available models count: {len(models)}")
        print("\n🔍 Checking for specific high-end models:")
        for target in ["gpt-4", "gpt-4o", "o1", "o1-preview", "o1-mini", "gpt-4.5"]:
            matches = [m for m in models if target in m]
            if matches:
                print(f"  ✅ Found {target} variants: {', '.join(matches)}")
            else:
                print(f"  ❌ No {target} models found")

        print("\n📋 First 20 models listed:")
        for m in sorted(models)[:20]:
            print(f"  - {m}")
    else:
        print(f"❌ Failed. Status Code: {response.status_code}")
        print("Error:", response.text)

except Exception as e:
    print(f"❌ Exception occurred: {e}")
