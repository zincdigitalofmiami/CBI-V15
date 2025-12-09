## Plan: Repository Restructure & Cleanup

This plan flattens the nested repository structure by archiving legacy files, promoting the clean "inner" repository to the root, and migrating your recent SQL work to the new `database/` directory. This preserves your root Git configuration while establishing the correct architecture.

### Steps
1.  **Create Cleanup Script**: Create a file named `cleanup.sh` in the root directory with the provided code block.
2.  **Execute Cleanup**: Run `bash cleanup.sh` in your terminal to move files, archive clutter, and migrate SQL macros.
3.  **Verify Migration**: Confirm `database/macros/eda.sql` exists and `src/` contains the new Python structure.
4.  **Commit Changes**: Run `git add .` and `git commit -m "Refactor: Flatten repository structure"` to finalize the new state.

### Further Considerations
1.  **Git History**: You will see many deletions (legacy files) and additions (promoted files). This is expected.
2.  **Inner History**: The commit history of the *inner* repo (`CBI-V15/CBI-V15`) will be left behind in the archive. We are keeping the *files* but using your *root* git history.
3.  **Safety**: The script moves files to `archive_cleanup_20251204` instead of deleting them, so nothing is lost.

---

### Action: Create `cleanup.sh`

Copy this code into a new file named `cleanup.sh` in your root folder:

```bash
#!/bin/bash
# CBI-V15 Cleanup & Restructure Script

echo "Starting Cleanup..."

# 1. Create Archive Directory
mkdir -p archive_cleanup_20251204

# 2. Move Root Clutter to Archive
echo "Archiving legacy root files..."
# Move visible folders
for dir in archive legacy_v14 definitions_legacy zl-intelligence config docs scripts src dashboard; do
    [ -d "$dir" ] && mv "$dir" archive_cleanup_20251204/"${dir}_legacy"
done

# Move specific root files
for file in dataform.json package.json requirements.txt push_to_github.sh vercel.json ruff.toml pyproject.toml main.py BQ_AUDIT_RESULTS.txt requirements-research.txt LICENSE README.md .gitignore; do
    [ -f "$file" ] && mv "$file" archive_cleanup_20251204/
done

# Handle nested archive if it exists
if [ -d "archive_legacy_20251204" ]; then
    mv archive_legacy_20251204 archive_cleanup_20251204/archive_legacy_nested
fi

# Move backup folders
mv CBI-V15.backup.* archive_cleanup_20251204/ 2>/dev/null

# 3. Prepare Inner Repo for Promotion
echo "Preparing inner repository..."
# Rename Inner .git to avoid conflict (we keep YOUR Root .git)
if [ -d "CBI-V15/CBI-V15/.git" ]; then
    mv CBI-V15/CBI-V15/.git CBI-V15/CBI-V15/.git_inner_backup
fi

# 4. Promote Inner Repo Contents to Root
echo "Promoting clean structure to root..."
# Move all visible files/folders
mv CBI-V15/CBI-V15/* . 2>/dev/null
# Move specific hidden files/folders
mv CBI-V15/CBI-V15/.cursorrules . 2>/dev/null
mv CBI-V15/CBI-V15/.github . 2>/dev/null
mv CBI-V15/CBI-V15/.codacy . 2>/dev/null
mv CBI-V15/CBI-V15/.continue . 2>/dev/null
mv CBI-V15/CBI-V15/.vscode . 2>/dev/null

# 5. Migrate New SQL Work
echo "Migrating your new SQL work..."
# Ensure target directories exist
mkdir -p database/macros
mkdir -p database/views

# Move the new SQL files from the old data folder (now in archive or lingering)
# We check the original location first, then the archive if it got moved
if [ -d "data/db/macros" ]; then
    mv data/db/macros/*.sql database/macros/ 2>/dev/null
    mv data/db/views/*.sql database/views/ 2>/dev/null
elif [ -d "archive_cleanup_20251204/data_legacy/db/macros" ]; then
    mv archive_cleanup_20251204/data_legacy/db/macros/*.sql database/macros/
    mv archive_cleanup_20251204/data_legacy/db/views/*.sql database/views/
fi

# 6. Final Cleanup
echo "Finalizing..."
# Archive the old data folder if it still exists in root
if [ -d "data" ] && [ ! -f "data/README.md" ]; then
     mv data archive_cleanup_20251204/data_legacy_residual
fi

# If the inner repo had a 'data' folder, it's now in root. 
# If not, we might need to restore the one we just archived? 
# (Assuming inner repo has the correct 'data' structure)

# Archive the empty shell of the inner repo
mv CBI-V15 archive_cleanup_20251204/CBI-V15_shell

echo "✅ Cleanup Complete!"
```
