#!/bin/bash

# Automated Evaluation Runner
# ==========================
# Simple bash script to run all model evaluations

echo "🚀 Starting automated evaluation of all models..."
echo "📍 Current directory: $(pwd)"

# Check if we're in the right directory
if [[ ! -d "LM" ]] || [[ ! -d "NLU" ]]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "Expected structure: /Users/marcoprosperi/Desktop/257857_Marco_Prosperi/"
    exit 1
fi

# Check if Python script exists
if [[ ! -f "run_all_evaluations.py" ]]; then
    echo "❌ Error: run_all_evaluations.py not found in current directory"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda command not found. Please make sure conda is installed and in your PATH."
    exit 1
fi

# Check if nlu25 environment exists
if ! conda env list | grep -q "nlu25"; then
    echo "❌ Error: conda environment 'nlu25' not found."
    echo "Please create the environment first: conda create -n nlu25 python=3.8"
    exit 1
fi

# Activate conda environment and run the Python evaluation script
echo "🔄 Activating conda environment 'nlu25'..."
echo "▶️ Running Python evaluation script..."

# Use conda run to ensure proper environment activation
conda run -n nlu25 python run_all_evaluations.py

# Check exit code
if [[ $? -eq 0 ]]; then
    echo "✅ Evaluation completed successfully!"
else
    echo "❌ Evaluation failed or was interrupted"
    exit 1
fi
