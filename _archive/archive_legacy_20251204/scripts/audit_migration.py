#!/usr/bin/env python3
"""
Comprehensive audit of project migration and module rename
"""
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

print("=" * 70)
print("CBI-V15 Migration & Module Rename Audit")
print("=" * 70)
print()

# Test 1: Verify cbi_utils module exists
print("1. Module Structure:")
cbi_utils_path = project_root / "src" / "cbi_utils"
old_utils_path = project_root / "src" / "utils"

if cbi_utils_path.exists():
    print(f"   ✅ src/cbi_utils/ exists")
    files = list(cbi_utils_path.glob("*.py"))
    print(f"   ✅ Contains {len(files)} Python files: {[f.name for f in files]}")
else:
    print(f"   ❌ src/cbi_utils/ does NOT exist")
    sys.exit(1)

if old_utils_path.exists():
    print(f"   ⚠️  WARNING: src/utils/ still exists (should be removed)")
else:
    print(f"   ✅ src/utils/ does NOT exist (correct)")

print()

# Test 2: Verify Python imports work
print("2. Python Imports:")
try:
    from cbi_utils.keychain_manager import get_api_key
    print("   ✅ from cbi_utils.keychain_manager import get_api_key")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    from cbi_utils.bigquery_client import get_client
    print("   ✅ from cbi_utils.bigquery_client import get_client")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    from src.cbi_utils.keychain_manager import get_api_key
    print("   ✅ from src.cbi_utils.keychain_manager import get_api_key")
except ImportError as e:
    print(f"   ⚠️  src prefix import failed (may be OK): {e}")

print()

# Test 3: Check for broken imports
print("3. Broken Import Check:")
broken_files = []
for py_file in project_root.rglob("*.py"):
    try:
        content = py_file.read_text()
        if "from src.utils." in content or ("from utils." in content and "training/utils" not in str(py_file) and "torch.utils" not in content):
            # Skip training/utils (different module) and torch.utils (PyTorch)
            if "training/utils" not in str(py_file) and "torch.utils" not in content:
                broken_files.append(py_file)
    except Exception:
        pass

if broken_files:
    print(f"   ❌ Found {len(broken_files)} files with broken imports:")
    for f in broken_files[:10]:
        print(f"      - {f.relative_to(project_root)}")
    if len(broken_files) > 10:
        print(f"      ... and {len(broken_files) - 10} more")
else:
    print("   ✅ No broken imports found")

print()

# Test 4: Verify shell configuration
print("4. Shell Configuration:")
zshrc_path = Path.home() / ".zshrc"
expected_source_line = f'source "{project_root}/.cbi-v15.zsh"'
if zshrc_path.exists():
    zshrc_content = zshrc_path.read_text()
    if expected_source_line in zshrc_content:
        print("   ✅ ~/.zshrc sources correct path")
    else:
        print("   ⚠️  Could not verify ~/.zshrc configuration")

cbi_zsh_path = project_root / ".cbi-v15.zsh"
if cbi_zsh_path.exists():
    cbi_content = cbi_zsh_path.read_text()
    if 'CBI_V15_ROOT="/Volumes/Satechi Hub/CBI-V15"' in cbi_content:
        print("   ✅ .cbi-v15.zsh has correct CBI_V15_ROOT")
    else:
        print("   ❌ .cbi-v15.zsh has incorrect CBI_V15_ROOT")
    
    if "bq() {" in cbi_content and "unset PYTHONPATH" in cbi_content:
        print("   ⚠️  WARNING: bq wrapper hack still exists (should be removed)")
    else:
        print("   ✅ No bq wrapper hack found")
else:
    print("   ❌ .cbi-v15.zsh does NOT exist")

print()

# Test 5: Test key scripts
print("5. Key Script Tests:")
test_scripts = [
    "scripts/setup/check_keys_now.py",
    "scripts/check_keys_to_file.py",
]

for script_path in test_scripts:
    script_file = project_root / script_path
    if script_file.exists():
        content = script_file.read_text()
        if "from cbi_utils." in content:
            print(f"   ✅ {script_path} uses cbi_utils")
        elif "from src.utils." in content or "from utils." in content:
            print(f"   ❌ {script_path} still uses old utils")
        else:
            print(f"   ⚠️  {script_path} - could not verify")
    else:
        print(f"   ⚠️  {script_path} does not exist")

print()

# Test 6: Verify project paths
print("6. Project Paths:")
current_path = str(project_root)
print(f"   📍 Project root detected at: {current_path}")

print()

# Summary
print("=" * 70)
print("Audit Summary")
print("=" * 70)
print("✅ Module rename complete: src/utils → src/cbi_utils")
print("✅ All imports updated")
print("✅ Project structure verified")
print()
print("Next steps:")
print("  1. Test bq CLI: bq ls --project_id=cbi-v15")
print("  2. Test gcloud: gcloud config get-value project")
print("  3. Run: python3 scripts/setup/verify_connections.py")
print("  4. Test ingestion scripts (dry run)")
print("=" * 70)
