#!/bin/bash
# Run integration tests without coverage requirements

echo "🔗 Running integration tests (mushroom, binary classification, end-to-end)..."
echo "📈 Capturing coverage data but no pass/fail requirements"
echo ""

./venv/bin/pytest -c pytest-integration.ini "$@"