# MotherDuck Quick Reference

## 🚀 Quick Setup (3 Steps)

```bash
# 1. Get tokens from Vercel Dashboard → Settings → Environment Variables
#    - motherduck_storage_MOTHERDUCK_TOKEN (bottom)
#    - motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN (top)

# 2. Run setup script
bash scripts/setup_motherduck_tokens.sh

# 3. Verify
python scripts/setup/verify_motherduck_tokens.py
```

## 📝 Manual Setup

### Option A: .env File (Recommended)
```bash
# Add to .env in project root
MOTHERDUCK_TOKEN=your_token_here
MOTHERDUCK_READ_SCALING_TOKEN=your_read_token_here
MOTHERDUCK_DB=cbi_v15
```

### Option B: Shell Environment
```bash
# Add to ~/.bashrc or ~/.zshrc
export MOTHERDUCK_TOKEN="your_token_here"
export MOTHERDUCK_READ_SCALING_TOKEN="your_read_token_here"
export MOTHERDUCK_DB="cbi_v15"

# Reload
source ~/.bashrc  # or source ~/.zshrc
```

## ✅ Verification

```bash
# Quick test
python scripts/test_motherduck_connection.py

# Full verification
python scripts/setup/verify_motherduck_tokens.py

# Manual test
python -c "from src.utils.motherduck_client import get_motherduck_connection; conn = get_motherduck_connection(); print('✅ Connected:', conn.execute('SELECT 1').fetchone())"
```

## 💻 Usage in Code

```python
# Recommended: Use centralized client
from src.utils.motherduck_client import get_motherduck_connection

conn = get_motherduck_connection()
result = conn.execute("SELECT * FROM raw.databento_futures_ohlcv_1d LIMIT 5").df()
```

## 🔍 Token Sources (Priority Order)

1. `MOTHERDUCK_TOKEN` (environment variable)
2. `motherduck_storage_MOTHERDUCK_TOKEN` (Vercel format)
3. `.env` file (project root)

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "MOTHERDUCK_TOKEN not found" | Check `.env` exists, verify token is set |
| "Could not connect" | Verify token validity, check network |
| "Invalid token format" | Remove quotes, check for whitespace |

## 📚 Full Documentation

See `docs/ops/MOTHERDUCK_SETUP.md` for complete details.
