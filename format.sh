#!/bin/bash

# Ensure the script runs in the project root
PROJECT_ROOT="$(dirname "$(realpath "$0")")"
SRC_DIR="$PROJECT_ROOT/src"

echo "Applying Black and isort formatting to all Python files in $SRC_DIR..."
echo "Using configuration from pyproject.toml in $PROJECT_ROOT"

# Check if Black and isort are installed
if ! command -v black &> /dev/null
then
    echo "Error: Black is not installed. Install it with: pip install black"
    exit 1
fi

if ! command -v isort &> /dev/null
then
    echo "Error: isort is not installed. Install it with: pip install isort"
    exit 1
fi

# Run isort first (sorts imports)
echo "Running isort..."
isort "$SRC_DIR" --settings-file "$PROJECT_ROOT/pyproject.toml"

# Run Black (formats code)
echo "Running Black..."
black "$SRC_DIR" --config "$PROJECT_ROOT/pyproject.toml"

echo "Formatting complete! 🎉"
