#!/bin/bash
# Run all tests with coverage capture but no fail requirements

echo "🧪🔗 Running ALL tests (unit + integration)..."
echo "📊 Capturing coverage data from all tests"
echo "⚠️  No coverage pass/fail requirements"
echo ""

./venv/bin/pytest "$@"