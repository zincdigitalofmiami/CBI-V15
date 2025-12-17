# OpenAI (GPT-5.1) Setup

Use your own OpenAI account and keep keys out of code.

## 1) Store the key

- Recommended: `./scripts/setup/store_api_keys.sh` (saves `OPENAI_API_KEY` to macOS Keychain and/or Secret Manager)
- Manual (Keychain): `security add-generic-password -a openai -s OPENAI_API_KEY -w YOUR_KEY -U`
- Manual (Secret Manager): `echo -n "YOUR_KEY" | gcloud secrets create openai-api-key --data-file=- --project=cbi-v15 --replication-policy=automatic`

## 2) Default model (optional)

- By default, helpers use `gpt-5.1`.
- To use Pro everywhere: `export OPENAI_MODEL=gpt-5.1-pro`

## 3) Use in code

```python
from src.utils.openai_client import run_chat, get_client, get_default_model

# Simple chat call
text = run_chat("Summarize today's ZL drivers", system="Keep it concise.")

# Or use the raw client
client = get_client()
resp = client.chat.completions.create(
    model=get_default_model(),  # respects OPENAI_MODEL if set
    messages=[{"role": "user", "content": "Ping"}],
)
```

## 4) Retrieve the key later

- Keychain: `security find-generic-password -s OPENAI_API_KEY -w`
- Secret Manager: `gcloud secrets versions access latest --secret=openai-api-key --project=cbi-v15`

## Gemini option (for Continue in VS Code)

- Export your Gemini key (do not commit it): `export GOOGLE_API_KEY="YOUR_GEMINI_KEY"`
- Workspace ships with `.continue/config.json` set to Gemini 3 Pro Preview by default:
  - `continue.provider`: `gemini`
  - `continue.model`: `models/gemini-3-pro-preview`
- Switch back to OpenAI by choosing the GPT-5.1 Pro model in Continue’s model picker (OpenAI key via `OPENAI_API_KEY`).
