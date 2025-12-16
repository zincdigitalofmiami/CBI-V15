# ⚠️ SECURITY ALERT: Token Rotation Required

**Date:** December 15, 2025
**Severity:** HIGH
**Action Required:** IMMEDIATE

---

## Issue

**MOTHERDUCK_TOKEN was exposed** during database audit session.

**Exposure Locations:**
- ✅ Chat history (AI assistant session)
- ✅ Terminal output from audit scripts
- ✅ Initial audit report draft

**Risk:**
- Exposed token could allow unauthorized access to MotherDuck database
- Token has **read/write** permissions

---

## Immediate Actions Required

### 1. Rotate MOTHERDUCK_TOKEN (Now)

```bash
# 1. Go to MotherDuck dashboard
open https://app.motherduck.com/

# 2. Navigate to: Settings → API Tokens
# 3. Revoke current token
# 4. Create new token with read/write permissions
# 5. Copy new token to clipboard
```

### 2. Update .env File

```bash
# Edit .env file
nano /Volumes/Satechi\ Hub/CBI-V15/.env

# Update MOTHERDUCK_TOKEN line:
MOTHERDUCK_TOKEN=<paste_new_token_here>

# Save and exit (Ctrl+X, Y, Enter)
```

### 3. Update Vercel Environment Variable

```bash
# Update Vercel environment variable
vercel env rm MOTHERDUCK_TOKEN
vercel env add MOTHERDUCK_TOKEN
# Paste new token when prompted

# Redeploy to apply new token
vercel --prod
```

### 4. Update macOS Keychain (if using)

```bash
# Update in Keychain Access app
open -a "Keychain Access"

# Search for "MotherDuck"
# Double-click credential
# Update password field with new token
```

### 5. Test New Token

```bash
# Test MotherDuck connection with new token
export MOTHERDUCK_TOKEN=<new_token_from_env>
python scripts/test_motherduck_connection.py

# Expected output:
# ✅ MotherDuck connection successful
# ✅ Database 'cbi_v15' accessible
```

---

## Prevention Measures (Going Forward)

### 1. NEVER Expose Tokens in Docs/Logs

**Always use placeholders:**
```
✅ GOOD: jdbc:duckdb:md:cbi_v15?motherduck_token=<YOUR_TOKEN>
❌ BAD:  jdbc:duckdb:md:cbi_v15?motherduck_token=eyJhbGc...actual_token
```

### 2. Configure Scripts to Mask Tokens

Update audit scripts to mask sensitive values:

```python
# Before printing connection string
conn_string = f"md:cbi_v15?motherduck_token={token}"
safe_string = conn_string.replace(token, "<REDACTED>")
print(f"Connection: {safe_string}")
```

### 3. Use Environment Variables Only

**Token storage hierarchy (in order of preference):**
1. `.env` file (gitignored) ✅
2. macOS Keychain ✅
3. IDE secure storage (IntelliJ Password Safe) ✅
4. Environment variable (session-only) ✅
5. **NEVER** hardcode in scripts ❌
6. **NEVER** commit to git ❌
7. **NEVER** paste in docs/tickets ❌

### 4. Verify .gitignore

```bash
# Ensure .env is gitignored
grep -E "^\.env$|^\.env\.local$" .gitignore

# Expected output:
.env
.env.local

# If missing, add:
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
```

### 5. Configure IntelliJ Data Sources Securely

**IntelliJ IDEA data source configuration:**
- ✅ Store in workspace (`.idea/dataSources.xml`) - **NOT committed**
- ✅ Enable "Password Safe" for sensitive fields
- ✅ Use `${MOTHERDUCK_TOKEN}` variable reference
- ❌ Never commit `.idea/dataSources.xml` to git

**Verify in `.gitignore`:**
```bash
grep -E "\.idea/dataSources\.xml|\.idea/dataSources/|\.idea/dataSources\.local\.xml" .gitignore

# Expected output:
.idea/dataSources.xml
.idea/dataSources.local.xml
.idea/dataSources/
```

---

## Token Rotation Checklist

- [ ] Rotate MOTHERDUCK_TOKEN in MotherDuck dashboard
- [ ] Update `.env` file with new token
- [ ] Update Vercel environment variable
- [ ] Update macOS Keychain (if used)
- [ ] Test new token with `test_motherduck_connection.py`
- [ ] Verify `.env` is in `.gitignore`
- [ ] Verify IntelliJ data sources are not committed
- [ ] Update any CI/CD secrets (if applicable)
- [ ] Document token rotation date in this file

---

## Token Rotation Log

| Date | Reason | Rotated By |
|------|--------|------------|
| 2025-12-15 | Token exposed in audit session | Chris (US Oil Solutions) |
| | | |
| | | |

---

## Contact

**Security Issues:** Chris (US Oil Solutions)
**MotherDuck Support:** support@motherduck.com
**Documentation:** `docs/setup/SECRETS_MANAGEMENT.md`

---

**Next Review:** 2026-03-15 (Quarterly token rotation)
**Status:** ⚠️ **ACTION REQUIRED** - Rotate token immediately
