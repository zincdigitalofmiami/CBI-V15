#!/usr/bin/env python3
"""
Wrapper script that runs commands and saves output to a file
This works around terminal output capture issues
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, output_file=None):
    """Run a command and optionally save output to file"""
    if output_file is None:
        output_file = Path.home() / ".cbi_v15_last_output.txt"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'command': cmd
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(f"Command: {cmd}\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write("=" * 60 + "\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n" + "=" * 60 + "\n")
            f.write("STDERR:\n")
            f.write(result.stderr)
            f.write("\n" + "=" * 60 + "\n")
        
        # Also print to stdout
        print(f"Command: {cmd}")
        print(f"Return Code: {result.returncode}")
        print("=" * 60)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print("=" * 60)
        print(f"\nFull output saved to: {output_file}")
        
        return output
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out after 30 seconds")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_with_output.py '<command>'")
        sys.exit(1)
    
    cmd = " ".join(sys.argv[1:])
    run_command(cmd)






