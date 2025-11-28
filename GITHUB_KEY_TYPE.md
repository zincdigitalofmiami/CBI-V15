# GitHub SSH Key Type: Authentication ✅

**Question**: Authentication or Signing?  
**Answer**: **AUTHENTICATION** ✅

---

## For Dataform: Use Authentication

When adding the RSA public key to GitHub, select **"Authentication Key"** (this is the default option).

### Why Authentication?

**Authentication** allows Dataform to:
- ✅ Connect to GitHub via SSH
- ✅ Clone/pull from the repository
- ✅ Push commits to the repository
- ✅ Access repository content

**Signing** is used for:
- ❌ Commit signing (GPG/SSH signing)
- ❌ Not needed for Dataform connection
- ❌ Different use case

---

## Steps to Add Key

1. **Go to**: https://github.com/settings/ssh/new

2. **Fill in the form**:
   - **Title**: `Dataform CBI-V15 RSA`
   - **Key**: Paste the RSA public key
   - **Key type**: Select **"Authentication Key"** (default) ✅

3. **Click**: "Add SSH key"

---

## Your RSA Public Key

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCwpwoBZQNgUwFT7d1e9HfH2z6KkGqzXL3yMlZhQKJXOtsufjOUhAziCABWHogZNSeE2v/LTEuSPSuT5mPM4wTqemAgnf3y75NIEUvxj6OfmNfWQjAA+DsIc7zHC9zvR9Dcr8IrOoS4OqTthy6YC/EqLyLCRhgZehML4Za7ywRrMl8oeDFJFJ04IzEyVhXNkRYFVBM43SwyOy3DgXw5/iHDDO+wVLdg/urPsKt7hSvnqfim8tb6Amp83ODH+MCukIQy+Wh3ulEIquToUhpEjziRkf98GRaZ2m/echdFAHoDh9+jEnU2rfeAJxuFnFKbEEL1DzwHttBG6rn/D9dEWM0Dp885wdPs/LqS3AjqOQFJFv/vtNbhRe6XBrMfj6x5uJW9PDQeSDlTJ7r8O13hgFFL88HY0Pkv8a/o7SIoT4wUyLJyV52F8gB/WP+FUO/Hu7uxcfejtbYlDK7mFZZcZpFEgO9pdlf+RgaJpMKqKFkBHWmMFw7Ye9t3AcYlfyDP7b6RJ3duBARl7nbpy7NRkgdiivK4o8PZmQIv+lF6B+rOL7o2T0Hbb771iWHVPhXkTa47WImpDIHH9CegfEgmYGrUR8VbuhM/JQF8COHAnovY/xB1qyIb8IdoE2ktIp3eA6dK2Jsl4N2Pf20HJMnzfRvAIGam9aAJoM42v42W5B1KRw== dataform-cbi-v15@gcp
```

---

## Verification

After adding, test the connection:

```bash
ssh -T git@github.com -i ~/.ssh/dataform_github_rsa
```

Should show: "Hi zincdigitalofmiami! You've successfully authenticated..."

---

## Summary

**Select**: ✅ **Authentication Key** (default option)

This is what Dataform needs to connect to GitHub and manage the repository.

---

**Status**: Ready to add key with **Authentication** type.

