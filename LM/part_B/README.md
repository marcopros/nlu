## Part 1.B: Advanced Regularisation Techniques
For the part 1.B of the project, I've implemented advanced techniques to the Language Model, such as Weight Tying, Variational Dropout and the non-monotonically triggered AvSGD.

### Notes:
To configure and run the code:

1. **Model Configuration**: Change the `MODEL_CONFIG` variable in `main.py` to select the desired model:
   - `"BASE"`: LSTM + Weight Tying
   - `"VARDROP"`: LSTM + Weight Tying + Variational Dropout  
   - `"FULL"`: LSTM + Weight Tying + Variational Dropout + AvSGD

2. **Evaluation Mode**: Set `EVALUATION_MODE = True` in `main.py` to evaluate a pre-trained model instead of training. Update `EVALUATION_MODEL_PATH` with the path to your saved model weights:
   - `"LM/part_B/bin/LSTM_WT/weights.pt"` (for BASE configuration)
   - `"LM/part_B/bin/LSTM_WT_VD/weights.pt"` (for VARDROP configuration)
   - `"LM/part_B/bin/LSTM_WT_VD_avSGD/weights.pt"` (for FULL configuration)

3. **Dataset Path**: If necessary, change the path of the dataset in `main.py` (lines starting with `train_raw`, `dev_raw`, `test_raw`).