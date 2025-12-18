# MotherDuck Setup & Token Management

## Overview

CBI-V15 uses MotherDuck as the cloud data warehouse. All scripts connect to MotherDuck using secure tokens stored in environment variables or `.env` file.

## Token Sources

MotherDuck tokens can be provided from multiple sources (checked in priority order):

1. **Environment Variables** (highest priority)
   - `MOTHERDUCK_TOKEN` - Primary read/write token
   - `MOTHERDUCK_READ_SCALING_TOKEN` - Read-only scaling token (for large queries)
   - `MOTHERDUCK_DB` - Database name (default: `cbi_v15`)

2. **Vercel Storage Format** (for deployment)
   - `motherduck_storage_MOTHERDUCK_TOKEN` - Alternative token name
   - `motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN` - Alternative read scaling token

3. **`.env` File** (project root)
   - Automatically loaded by `src/utils/motherduck_client.py`
   - Format: `MOTHERDUCK_TOKEN=your_token_here`

## Required Tokens

### Primary Token (MOTHERDUCK_TOKEN)
- **Purpose**: Read/write access to MotherDuck database
- **Scope**: Full database operations (INSERT, UPDATE, DELETE, SELECT)
- **Source**: MotherDuck dashboard → Service Accounts → Create Token
- **Location in Vercel**: Storage → `motherduck_storage_MOTHERDUCK_TOKEN` (bottom value)

### Read Scaling Token (MOTHERDUCK_READ_SCALING_TOKEN)
- **Purpose**: Optimized read-only access for large queries
- **Scope**: SELECT operations only (no writes)
- **Source**: MotherDuck dashboard → Service Accounts → Create Read Scaling Token
- **Location in Vercel**: Storage → `motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN` (top value)

### Database Name (MOTHERDUCK_DB)
- **Default**: `cbi_v15`
- **Purpose**: Specifies which MotherDuck database to connect to

## Setup Instructions

### Option 1: Using .env File (Recommended for Local Development)

1. **Create/Edit `.env` file** in project root:
   ```bash
   cd /Volumes/Satechi\ Hub/CBI-V15
   ```

2. **Add tokens to `.env`**:
   ```bash
   # Primary read/write token
   MOTHERDUCK_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   
   # Read scaling token (optional, for large read queries)
   MOTHERDUCK_READ_SCALING_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   
   # Database name
   MOTHERDUCK_DB=cbi_v15
   ```

3. **Verify setup**:
   ```bash
   python scripts/setup/verify_motherduck_tokens.py
   ```

### Option 2: Using Shell Environment Variables

1. **Add to shell config** (`~/.bashrc` or `~/.zshrc`):
   ```bash
   export MOTHERDUCK_TOKEN="your_token_here"
   export MOTHERDUCK_READ_SCALING_TOKEN="your_read_scaling_token_here"
   export MOTHERDUCK_DB="cbi_v15"
   ```

2. **Reload shell config**:
   ```bash
   source ~/.bashrc  # or source ~/.zshrc
   ```

3. **Verify**:
   ```bash
   echo $MOTHERDUCK_TOKEN  # Should show your token
   python scripts/setup/verify_motherduck_tokens.py
   ```

### Option 3: Automated Setup from Vercel

If you have tokens in Vercel storage, use the setup script:

```bash
bash scripts/setup_motherduck_tokens.sh
```

This script will:
1. Prompt for tokens from Vercel
2. Add them to your shell config (`~/.bashrc` or `~/.zshrc`)
3. Test the connection

## Token Retrieval from Vercel

If tokens are stored in Vercel:

1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Find these variables:
   - `motherduck_storage_MOTHERDUCK_TOKEN` (bottom one - read/write)
   - `motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN` (top one - read-only)
3. Click the **Copy** icon (squares) or **Eye** icon to reveal/copy values
4. Paste into `.env` file or run setup script

## Verification

### Quick Test
```bash
python scripts/test_motherduck_connection.py
```

### Full Verification
```bash
python scripts/setup/verify_motherduck_tokens.py
```

### Manual Test
```python
from src.utils.motherduck_client import get_motherduck_connection

conn = get_motherduck_connection()
result = conn.execute("SELECT 1 as test").fetchone()
print(f"✅ Connected: {result}")
```

## Usage in Scripts

### Recommended: Use Centralized Client

All Python scripts should use the centralized client:

```python
from src.utils.motherduck_client import get_motherduck_connection

# Standard connection (read/write)
conn = get_motherduck_connection()

# Read-only connection (uses read scaling token if available)
conn = get_motherduck_connection(read_only=True, use_read_scaling=True)

# Execute queries
result = conn.execute("SELECT * FROM raw.databento_futures_ohlcv_1d LIMIT 5").df()
```

### Legacy: Direct Connection (Not Recommended)

If you must connect directly:

```python
import os
import duckdb
from dotenv import load_dotenv

load_dotenv()  # Load .env file

token = os.getenv("MOTHERDUCK_TOKEN")
db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")

conn = duckdb.connect(f"md:{db_name}?motherduck_token={token}")
```

## Troubleshooting

### Error: "MOTHERDUCK_TOKEN not found"

**Solution**: 
1. Check `.env` file exists in project root
2. Verify token is set: `echo $MOTHERDUCK_TOKEN`
3. Ensure `.env` is not in `.gitignore` (it should be)
4. Try: `python scripts/setup/verify_motherduck_tokens.py`

### Error: "Could not connect to MotherDuck"

**Possible causes**:
1. Token expired or invalid
2. Network connectivity issues
3. Token doesn't have required permissions

**Solution**:
1. Verify token in MotherDuck dashboard
2. Generate new token if expired
3. Check network connection
4. Test with: `python scripts/test_motherduck_connection.py`

### Error: "Invalid token format"

**Solution**:
- Ensure token doesn't have extra quotes: `MOTHERDUCK_TOKEN=token` (not `MOTHERDUCK_TOKEN="token"`)
- Remove any trailing whitespace
- Check token wasn't truncated when copying

## Security Best Practices

1. **Never commit tokens to git**
   - `.env` should be in `.gitignore`
   - Never hardcode tokens in scripts
   - Use environment variables or `.env` file

2. **Use read-only tokens when possible**
   - For read-only operations, use `MOTHERDUCK_READ_SCALING_TOKEN`
   - Reduces risk if token is compromised

3. **Rotate tokens regularly**
   - Generate new tokens every 90 days
   - Update `.env` and shell configs
   - Update Vercel environment variables

4. **Use service accounts**
   - Create dedicated service accounts for different operations
   - Limit token permissions to minimum required

## Token Locations Reference

| Location | Variable Name | Purpose |
|----------|--------------|---------|
| `.env` | `MOTHERDUCK_TOKEN` | Primary token (read/write) |
| `.env` | `MOTHERDUCK_READ_SCALING_TOKEN` | Read scaling token |
| `.env` | `MOTHERDUCK_DB` | Database name |
| `~/.bashrc` or `~/.zshrc` | `MOTHERDUCK_TOKEN` | Shell environment |
| Vercel Storage | `motherduck_storage_MOTHERDUCK_TOKEN` | Deployment token |
| Vercel Storage | `motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN` | Deployment read token |

## Related Documentation

- `scripts/setup_motherduck_tokens.sh` - Automated token setup script
- `scripts/test_motherduck_connection.py` - Connection test script
- `scripts/setup/verify_motherduck_tokens.py` - Token verification script
- `src/utils/motherduck_client.py` - Centralized connection client
- `docs/ops/MOTHERDUCK_VERCEL_CONNECTION_AUDIT.md` - Vercel integration details

## Support

If you continue to have issues:
1. Run `python scripts/setup/verify_motherduck_tokens.py` for diagnostics
2. Check `docs/ops/MOTHERDUCK_VERCEL_CONNECTION_AUDIT.md` for Vercel-specific issues
3. Verify tokens in MotherDuck dashboard → Service Accounts
