#!/usr/bin/env python3
"""Simple test script to verify output works"""
import sys
import os

print("=" * 60)
print("OUTPUT TEST SCRIPT")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print("=" * 60)
print("✅ If you see this, output is working!")
sys.stdout.flush()
sys.stderr.write("STDERR test message\n")
sys.stderr.flush()






