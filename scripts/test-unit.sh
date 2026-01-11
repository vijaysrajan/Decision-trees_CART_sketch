#!/bin/bash
# Run unit tests with coverage requirements

echo "🧪 Running unit tests with 90% coverage requirement..."
echo "📊 Excluding integration tests marked with @pytest.mark.integration"
echo ""

./venv/bin/pytest -c pytest-unit.ini "$@"