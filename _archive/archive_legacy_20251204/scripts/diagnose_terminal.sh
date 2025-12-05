#!/bin/bash
# Terminal output diagnostic script

echo "=== TERMINAL DIAGNOSTIC TEST ===" >&2
echo "STDOUT_TEST_12345"
echo "STDERR_TEST_12345" >&2

# Test Python output
python3 -c "import sys; print('PYTHON_STDOUT'); sys.stderr.write('PYTHON_STDERR\n')"

# Test with explicit flush
python3 -c "import sys; sys.stdout.write('FLUSHED_OUTPUT\n'); sys.stdout.flush()"

# Test environment
echo "SHELL: $SHELL"
echo "PWD: $(pwd)"

# Test keychain check
echo "Testing keychain access..."
security find-generic-password -s "DATABENTO_API_KEY" -w 2>&1 | head -1 || echo "Keychain test: No key found (this is OK)"

echo "=== END DIAGNOSTIC ===" >&2






