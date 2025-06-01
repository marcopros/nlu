<p align='center'>
    <h1 align="center">Natural Language Understanding</h1>
    <p align="center">
    NLU Project @ University of Trento, Italy
    </p>
    <p align='center'>
      Prosperi Marco <br>
      MSc in AIS - 24/25
    </p>   
</p>


This repository contains implementations for **Language Modeling** and **Natural Language Understanding** tasks, featuring both traditional neural networks and transformer-based approaches.

## 📁 Project Structure

```
├── LM/                          # Language Modeling
│   ├── part_A/                  # Traditional RNN/LSTM approaches
│   ├── part_B/                  # Advanced techniques (Weight Tying, Variational Dropout)
│   └── report.pdf               # Detailed technical report
├── NLU/                         # Natural Language Understanding
│   ├── part_A/                  # BiLSTM for Joint Slot Filling & Intent Classification
│   ├── part_B/                  # BERT-based approaches
│   └── report.pdf               # Detailed technical report
├── run_all_evaluations.py       # Automated evaluation suite
├── run_evaluations.sh          # Bash wrapper for evaluations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**: Python 3.8+ required
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running All Evaluations (Recommended)

To test all trained models across all project parts:

```bash
# Option 1: Using bash script
./run_evaluations.sh

# Option 2: Direct Python execution  
python run_all_evaluations.py
```

## 📊 Project Components

### 🔤 Language Modeling (LM)

#### Part A: Traditional Approaches
**Location**: `LM/part_A/`

**Models Implemented**:
- **RNN Baseline**: Simple RNN architecture
- **LSTM**: Long Short-Term Memory networks
- **LSTM + Dropout**: LSTM with regularization
- **LSTM + Dropout + AdamW**: Advanced optimization

**Dataset**: Penn TreeBank

**Metric**: Perplexity (lower is better)

**How to Run**:
```bash
cd LM/part_A/

# For training
python main.py

# For evaluation (modify main.py first)
# Set EVALUATION_MODE = True
# Set MODEL_CONFIG to desired model ("RNN", "LSTM", "LSTM_DROPOUT", "LSTM_DROPOUT_ADAMW")
# Set EVALUATION_MODEL_PATH to corresponding weights file
python main.py
```

#### Part B: Advanced Techniques
**Location**: `LM/part_B/`

**Models Implemented**:
- **BASE**: LSTM + Weight Tying
- **VARDROP**: BASE + Variational Dropout
- **FULL**: VARDROP + Averaged SGD

**Dataset**: Penn TreeBank

**Metric**: Perplexity (lower is better)

**How to Run**:
```bash
cd LM/part_B/

# For training
python main.py

# For evaluation (modify main.py first)
# Set EVALUATION_MODE = True  
# Set MODEL_CONFIG to desired model ("BASE", "VARDROP", "FULL")
# Set EVALUATION_MODEL_PATH to corresponding weights file
python main.py
```

### 🗣️ Natural Language Understanding (NLU)

#### Part A: BiLSTM Joint Models
**Location**: `NLU/part_A/`

**Models Implemented**:
- **IAS_BASELINE**: Unidirectional LSTM
- **IAS_BIDIR**: Bidirectional LSTM
- **IAS_BIDIR_DROPOUT**: BiLSTM + Dropout

**Dataset**: ATIS (Airline Travel Information System)

**Metrics**: 
- Slot F1 Score (higher is better)
- Intent Accuracy (higher is better)

**How to Run**:
```bash
cd NLU/part_A/

# For training
python main.py

# For evaluation (modify main.py first)
# Set EVALUATION_MODE = True
# Set MODEL_CONFIG to desired model ("IAS_BASELINE", "IAS_BIDIR", "IAS_BIDIR_DROPOUT") 
# Set EVALUATION_MODEL_PATH to corresponding weights file
python main.py
```

#### Part B: BERT-based Approaches
**Location**: `NLU/part_B/`

**Models Implemented**:
- **BERT-base**: bert-base-uncased fine-tuned for joint task
- **BERT-large**: bert-large-uncased fine-tuned for joint task

**Dataset**: ATIS (Airline Travel Information System)

**Metrics**:
- Slot F1 Score (higher is better)  
- Intent Accuracy (higher is better)

**How to Run**:
```bash
cd NLU/part_B/

# For training
python main.py

# For evaluation (modify main.py first)
# Set EVALUATION_MODE = True
# Set EVALUATION_MODEL_PATH to desired weights file:
# - "NLU/part_B/bin/bert-base/weights.pt" (for BERT-base)
# - "NLU/part_B/bin/bert-large/weights.pt" (for BERT-large)
python main.py
```

## 🔧 Configuration Guide

Each project part follows a similar configuration pattern in `main.py`:

### Common Configuration Variables:

1. **EVALUATION_MODE**: 
   - `True`: Load pre-trained model and evaluate
   - `False`: Train new model from scratch

2. **MODEL_CONFIG**: Select which model variant to use (varies by project part)

3. **EVALUATION_MODEL_PATH**: Path to saved model weights for evaluation

4. **Dataset Paths**: Modify if datasets are in different locations

### Example Configuration (NLU Part B):
```python
# In main.py
EVALUATION_MODE = True  # Set to True for evaluation
EVALUATION_MODEL_PATH = "NLU/part_B/bin/bert-base/weights.pt"
```

## 📈 Automated Evaluation Suite

The automated evaluation suite (`run_all_evaluations.py`) provides:

### Features:
- **Auto-Configuration**: Automatically modifies each `main.py` for evaluation mode
- **Safe Execution**: Creates backups and restores original files
- **Result Extraction**: Parses outputs to extract key metrics
- **Comprehensive Summary**: Displays formatted results for all models

### Sample Output:
```
================================================================================
                    📊 COMPREHENSIVE EVALUATION SUMMARY                    
================================================================================

🔤 Language Modeling - Part A (Perplexity ↓)
   RNN                 :  123.45
   LSTM                :   98.76
   LSTM_DROPOUT        :   87.65
   LSTM_DROPOUT_ADAMW  :   82.34
   🏆 Best: LSTM_DROPOUT_ADAMW (PPL: 82.34)

🗣️ NLU - Part B (F1 Score ↑ | Intent Acc ↑)
   bert-base-uncased   : F1=0.9547 | Acc=0.9686
   bert-large-uncased  : F1=0.9634 | Acc=0.9721
   🏆 Best F1: bert-large-uncased (F1: 0.9634)
```

### Execution Time:
- **Total Runtime**: 5-15 minutes (depending on hardware)
- **Individual Models**: 30 seconds to 2 minutes each
- **BERT Models**: Up to 10 minutes for BERT-large

## 🛠️ Troubleshooting

### Common Issues:

1. **Missing Model Files**:
   ```
   ⚠️ Model file not found: LM/part_A/bin/RNN_baseline/weights.pt
   ```
   **Solution**: Ensure all models have been trained and saved in expected locations

2. **Import Errors**:
   **Solution**: Run scripts from their respective directories or project root

3. **CUDA/GPU Issues**:
   **Solution**: Models automatically fall back to CPU if CUDA unavailable

4. **Version Compatibility**:
   - For BERT models, if you encounter missing keys (e.g., `position_ids`), the code handles this automatically with `strict=False` loading

### Manual Model Evaluation:

If you want to evaluate a specific model manually:

```bash
# Navigate to the appropriate directory
cd LM/part_A/  # or LM/part_B/, NLU/part_A/, NLU/part_B/

# Edit main.py to set:
# EVALUATION_MODE = True
# MODEL_CONFIG = "desired_model"  # (for LM and NLU part A)
# EVALUATION_MODEL_PATH = "path/to/weights.pt"

# Run evaluation
python main.py
```

## 📝 File Structure Details

### Key Files in Each Part:

- **`main.py`**: Main execution script with configuration options
- **`model.py`**: Neural network architecture definitions
- **`functions.py`**: Training and evaluation functions
- **`utils.py`**: Data processing and utility functions
- **`bin/`**: Directory containing trained model weights
- **`dataset/`**: Directory containing training/test data

### Model Weights Locations:

```
LM/part_A/bin/
├── RNN_baseline/weights.pt
├── LSTM/weights.pt
├── LSTM_Drop/weights.pt
└── LSTM_Drop_AdamW/weights.pt

LM/part_B/bin/
├── LSTM_WT/weights.pt
├── LSTM_WT_VD/weights.pt
└── LSTM_WT_VD_avSGD/weights.pt

NLU/part_A/bin/
├── IAS_BASELINE/weights_1.pt
├── IAS_BIDIR/weights_1.pt
└── IAS_BIDIR_DROPOUT/weights_1.pt

NLU/part_B/bin/
├── bert-base/weights.pt
└── bert-large/weights.pt
```

## 🎯 Expected Results

### Language Modeling:
- **Part A**: Perplexity improvements from ~120 (RNN) to ~80 (LSTM+Dropout+AdamW)
- **Part B**: Further improvements to ~65 with advanced techniques

### NLU Tasks:
- **Part A**: F1 scores ~0.85-0.88, Intent accuracy ~0.91-0.93
- **Part B**: F1 scores ~0.95+, Intent accuracy ~0.96+ with BERT

## 📚 Additional Resources

- **Technical Reports**: Detailed analysis available in `LM/report.pdf` and `NLU/report.pdf`
- **Code Documentation**: Each module contains inline documentation
- **Dataset Information**: Standard Penn TreeBank (LM) and ATIS (NLU) datasets

## 🔍 Debugging

For verbose output during evaluation:
```bash
python run_all_evaluations.py 2>&1 | tee evaluation_log.txt
```

This saves all output to `evaluation_log.txt` for debugging purposes.

---

For questions or issues, please refer to the individual README files in each project part or check the technical reports for detailed methodology and results.
