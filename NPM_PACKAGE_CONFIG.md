# NPM Package Configuration Guide

**Question**: Should I configure a private npm package using the Dataform SSH key?

---

## Current SSH Key Status

**Key**: `~/.ssh/dataform_github_rsa`
- **Purpose**: Dataform ↔ GitHub connection
- **Format**: PEM (`-----BEGIN RSA PRIVATE KEY-----`)
- **On GitHub**: ✅ Yes (added as "Dataform CBI-V15 RSA")
- **In Secret Manager**: ✅ Yes (for Dataform service account)

---

## Options for Private NPM Packages

### Option A: Reuse Same SSH Key (GitHub Packages)

**If using GitHub Packages:**
- ✅ Can reuse the same SSH key
- ✅ Already on GitHub
- ⚠️  Shared key (less secure separation)

**Configuration:**
```bash
# Add to ~/.npmrc
@zincdigital:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=YOUR_GITHUB_TOKEN
```

**Or with SSH:**
```bash
# Configure Git to use SSH for GitHub Packages
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

---

### Option B: Create Separate SSH Key (Recommended)

**Benefits:**
- ✅ Better security (separation of concerns)
- ✅ Can revoke independently
- ✅ Different permissions if needed

**Create new key:**
```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/npm_github_rsa \
    -C "npm-packages-cbi-v15@gcp" -N ""

# Add to GitHub
cat ~/.ssh/npm_github_rsa.pub
# Add at: https://github.com/settings/ssh/new
```

**Configure npm:**
```bash
# Add to ~/.npmrc
@zincdigital:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=YOUR_GITHUB_TOKEN
```

---

### Option C: Use NPM Token (npm Registry)

**If using npm registry (not GitHub Packages):**
- ✅ Use npm token (not SSH)
- ✅ More standard for npm registry
- ✅ Easier to manage

**Configuration:**
```bash
# Login to npm
npm login

# Or set token directly
npm config set //registry.npmjs.org/:_authToken YOUR_NPM_TOKEN
```

---

## Recommendation

**For CBI-V15 project:**

1. **If using GitHub Packages**: 
   - ✅ **Reuse the Dataform SSH key** (already configured)
   - Or create separate key for better separation

2. **If using npm registry**:
   - ✅ **Use npm token** (not SSH key)
   - More appropriate for npm registry

3. **Best Practice**:
   - ✅ **Create separate SSH key** for npm/GitHub Packages
   - Better security separation
   - Can manage independently

---

## Which Are You Using?

**Please clarify:**
- [ ] GitHub Packages (npm.pkg.github.com)
- [ ] Private npm registry (registry.npmjs.org)
- [ ] Other private registry

**Then I can provide specific configuration steps.**

---

## Quick Setup (If GitHub Packages)

**Reuse existing key:**
```bash
# Already configured! Just need npm config
echo "@zincdigital:registry=https://npm.pkg.github.com" >> ~/.npmrc
echo "//npm.pkg.github.com/:_authToken=YOUR_GITHUB_TOKEN" >> ~/.npmrc
```

**Or create separate key:**
```bash
# Create new key
ssh-keygen -t rsa -b 4096 -f ~/.ssh/npm_github_rsa \
    -C "npm-packages-cbi-v15@gcp" -N ""

# Add public key to GitHub
cat ~/.ssh/npm_github_rsa.pub
# Add at: https://github.com/settings/ssh/new
```

---

**Let me know which option you prefer and I'll help configure it!**

