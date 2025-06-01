# Natural Language Understanding & Language Modeling Project

Implementation of Language Modeling and Natural Language Understanding tasks using traditional neural networks and transformer-based approaches.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all evaluations
./run_evaluations.sh
# or
python run_all_evaluations.py
```

## 📁 Project Overview

| Component | Task | Models | Dataset | Metric |
|-----------|------|--------|---------|--------|
| **LM/part_A** | Language Modeling | RNN, LSTM variants | Penn TreeBank | Perplexity ↓ |
| **LM/part_B** | Advanced LM | Weight Tying, Var. Dropout | Penn TreeBank | Perplexity ↓ |
| **NLU/part_A** | Joint Slot+Intent | BiLSTM variants | ATIS | F1, Accuracy ↑ |
| **NLU/part_B** | Joint Slot+Intent | BERT models | ATIS | F1, Accuracy ↑ |

## 🔧 Running Individual Models

### Manual Evaluation
1. Navigate to the model directory (`cd LM/part_A/`, `NLU/part_A/`, etc.)
2. Edit `main.py`:
   ```python
   EVALUATION_MODE = True
   MODEL_CONFIG = "model_name"  # See model lists below
   EVALUATION_MODEL_PATH = "path/to/weights.pt"
   ```
3. Run: `python main.py`

### Available Models

<details>
<summary><b>🔤 Language Modeling Models</b></summary>

**LM Part A** (`cd LM/part_A/`):
- `RNN` → `bin/RNN_baseline/weights.pt`
- `LSTM` → `bin/LSTM/weights.pt`
- `LSTM_DROPOUT` → `bin/LSTM_Drop/weights.pt`
- `LSTM_DROPOUT_ADAMW` → `bin/LSTM_Drop_AdamW/weights.pt`

**LM Part B** (`cd LM/part_B/`):
- `BASE` → `bin/LSTM_WT/weights.pt`
- `VARDROP` → `bin/LSTM_WT_VD/weights.pt`
- `FULL` → `bin/LSTM_WT_VD_avSGD/weights.pt`

</details>

<details>
<summary><b>🗣️ NLU Models</b></summary>

**NLU Part A** (`cd NLU/part_A/`):
- `IAS_BASELINE` → `bin/IAS_BASELINE/weights_1.pt`
- `IAS_BIDIR` → `bin/IAS_BIDIR/weights_1.pt`
- `IAS_BIDIR_DROPOUT` → `bin/IAS_BIDIR_DROPOUT/weights_1.pt`

**NLU Part B** (`cd NLU/part_B/`):
- BERT-base → `bin/bert-base/weights.pt`
- BERT-large → `bin/bert-large/weights.pt`

</details>

## 📈 Automated Evaluation Results

**Runtime**: 5-15 minutes total (individual models: 30s-2min, BERT-large: up to 10min)

**Sample Output**:
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

### Expected Results
- **LM Part A**: Perplexity ~120 (RNN) → ~80 (LSTM+Dropout+AdamW)
- **LM Part B**: Perplexity ~65 with advanced techniques
- **NLU Part A**: F1 ~0.85-0.88, Intent accuracy ~0.91-0.93
- **NLU Part B**: F1 ~0.95+, Intent accuracy ~0.96+ (BERT)

## 🛠️ Troubleshooting

<details>
<summary><b>Common Issues & Solutions</b></summary>

**Missing Model Files**:
```
⚠️ Model file not found: LM/part_A/bin/RNN_baseline/weights.pt
```
→ Ensure all models have been trained and saved

**Import/CUDA Errors**:
→ Run scripts from correct directories; models auto-fallback to CPU

**Version Compatibility**:
→ BERT models handle missing keys automatically with `strict=False`

**Debugging**:
```bash
python run_all_evaluations.py 2>&1 | tee evaluation_log.txt
```

</details>

## 📚 Additional Resources

- **Technical Reports**: `LM/report.pdf`, `NLU/report.pdf`
- **Datasets**: Penn TreeBank (LM), ATIS (NLU)
- **Code Documentation**: Inline documentation in each module

---

*For detailed methodology and results, refer to the technical reports in each project part.*
