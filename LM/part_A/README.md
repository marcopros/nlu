## Part 1.A: Improving the Baseline ```LM_RNN```

In this part of the project, I applied several modifications to the baseline ```LM_RNN``` model to improve its performance. Each modification was added incrementally, and its impact on Perplexity (PPL) was evaluated. If a modification resulted in worse performance, I removed it and moved on to the next one. 

### Note:
To configure and run the code:

1. **Model Configuration**: Change the `MODEL_CONFIG` variable in `main.py` to select the desired model:
   - `"RNN"`: Baseline RNN model
   - `"LSTM"`: LSTM model  
   - `"LSTM_DROPOUT"`: LSTM with dropout layers
   - `"LSTM_DROPOUT_ADAMW"`: LSTM with dropout layers and AdamW optimizer

2. **Evaluation Mode**: Set `EVALUATION_MODE = True` in `main.py` to evaluate a pre-trained model instead of training. Update `EVALUATION_MODEL_PATH` with the path to your saved model weights.

3. **Dataset Path**: If necessary, change the path of the dataset in `main.py` (lines starting with `train_raw`, `dev_raw`, `test_raw`).

