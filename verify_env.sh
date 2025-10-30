#!/bin/bash
# Script to verify development environment setup

set -e  # Exit on error

echo "=== Theta Sketch Tree Development Environment Verification ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment is activated or exists
if [ ! -d "venv" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo "Run: python3 -m venv venv"
    exit 1
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Check Python version
echo ""
echo "Checking Python version..."
PYTHON_VERSION=$(./venv/bin/python --version)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

# Check installed packages
echo ""
echo "Checking core dependencies..."
./venv/bin/python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
./venv/bin/python -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')"
./venv/bin/python -c "import scipy; print(f'  scipy: {scipy.__version__}')"
./venv/bin/python -c "import pandas; print(f'  pandas: {pandas.__version__}')"
./venv/bin/python -c "import yaml; print(f'  pyyaml: {yaml.__version__}')"
./venv/bin/python -c "import datasketches; print(f'  datasketches: installed')"
echo -e "${GREEN}✓ All core dependencies installed${NC}"

# Check development tools
echo ""
echo "Checking development tools..."
PYTEST_VERSION=$(./venv/bin/pytest --version | head -n1)
BLACK_VERSION=$(./venv/bin/black --version | head -n1)
FLAKE8_VERSION=$(./venv/bin/flake8 --version | head -n1)
MYPY_VERSION=$(./venv/bin/mypy --version)

echo "  $PYTEST_VERSION"
echo "  $BLACK_VERSION"
echo "  $FLAKE8_VERSION"
echo "  $MYPY_VERSION"
echo -e "${GREEN}✓ All development tools installed${NC}"

# Check configuration files
echo ""
echo "Checking configuration files..."
CONFIG_FILES=("pytest.ini" "pyproject.toml" ".flake8" ".gitignore")
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file (missing)"
    fi
done

# Run quick test discovery
echo ""
echo "Testing pytest configuration..."
TEST_COUNT=$(./venv/bin/pytest --collect-only -q 2>/dev/null | tail -n1 | grep -o '[0-9]\+' || echo "0")
echo -e "${GREEN}✓ Found $TEST_COUNT test files${NC}"

# Summary
echo ""
echo "=== Environment Verification Complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Available development commands:"
echo "  pytest                 # Run tests"
echo "  black theta_sketch_tree/   # Format code"
echo "  flake8 theta_sketch_tree/  # Check style"
echo "  mypy theta_sketch_tree/    # Type check"
