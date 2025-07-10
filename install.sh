#!/bin/bash
# Installation script for AlphabowRiskAssessmentV3

set -e

echo "============================================================"
echo "AlphabowRiskAssessmentV3 Installation Script"
echo "============================================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✓ Python $python_version detected (required: $required_version+)"
else
    echo "✗ Python $required_version or higher is required"
    echo "Please install Python $required_version+ and try again"
    exit 1
fi

# Check if pip is available
echo "Checking pip availability..."
if command -v pip3 &> /dev/null; then
    echo "✓ pip3 is available"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    echo "✓ pip is available"
    PIP_CMD="pip"
else
    echo "✗ pip is not available"
    echo "Please install pip and try again"
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    $PIP_CMD install -r requirements.txt --user
    echo "✓ Dependencies installed"
else
    echo "✗ requirements.txt not found"
    echo "Installing core dependencies manually..."
    $PIP_CMD install --user pandas numpy scipy openpyxl matplotlib seaborn pytest
    echo "✓ Core dependencies installed"
fi

# Create directories
echo "Creating necessary directories..."
mkdir -p results
mkdir -p logs
mkdir -p example_results
echo "✓ Directories created"

# Run tests
echo "Running tests to verify installation..."
if python3 -m pytest test_alphabow_risk_assessment.py -v --tb=short; then
    echo "✓ All tests passed"
else
    echo "⚠ Some tests failed, but installation may still work"
fi

# Run basic functionality test
echo "Running basic functionality test..."
if python3 test_alphabow_risk_assessment.py > /dev/null 2>&1; then
    echo "✓ Basic functionality test passed"
else
    echo "⚠ Basic functionality test had issues, check data files"
fi

# Check data files
echo "Checking for data files..."
required_files=("production_export.csv" "pipelines export data.csv")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "⚠ Missing: $file"
        missing_files+=("$file")
    fi
done

# Check optional files
optional_files=("pine_creek_cp_data.xlsx" "updated_well_list.csv")
for file in "${optional_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found (optional): $file"
    else
        echo "- Missing (optional): $file"
    fi
done

echo ""
echo "============================================================"
echo "Installation Summary"
echo "============================================================"

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✓ Installation completed successfully!"
    echo "✓ All required data files found"
    echo ""
    echo "You can now run the risk assessment tool:"
    echo "  python3 AlphabowRiskAssessmentV3.py"
    echo ""
    echo "Or run the example script:"
    echo "  python3 example_usage.py"
else
    echo "⚠ Installation completed with warnings"
    echo "Missing required data files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    fi
    echo ""
    echo "Please add the missing data files and run again."
fi

echo ""
echo "For help and documentation, see README.md"
echo "============================================================"