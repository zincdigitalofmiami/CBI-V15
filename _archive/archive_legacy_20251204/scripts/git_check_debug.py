import subprocess
import os

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout + "\n" + result.stderr
    except Exception as e:
        return str(e)

output = "=== GIT STATUS ===\n"
output += run_cmd("git status")

output += "\n=== CURRENT BRANCH ===\n"
branch = run_cmd("git branch --show-current").strip()
output += branch + "\n"

output += "\n=== UNPUSHED COMMITS ===\n"
if branch:
    output += run_cmd(f"git log origin/{branch}..HEAD --oneline")
else:
    output += "Could not determine branch."

with open("git_report.txt", "w") as f:
    f.write(output)



