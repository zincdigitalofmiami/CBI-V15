#!/bin/bash
# Force output by explicitly writing to /dev/tty
# This bypasses any stdout/stderr redirection

exec >/dev/tty 2>&1

echo "FORCE_OUTPUT_TEST_START"
echo "Current directory: $(pwd)"
echo "Shell: $SHELL"
echo "Testing Python..."
python3 -c "print('Python output test')"
echo "FORCE_OUTPUT_TEST_END"






