## Part 2.A : Slot filling and intent classification
### Model Architecture
-----------------------

This project implements a **joint BiLSTM-based model** for slot filling and intent classification on natural language utterances.

### Components

- **Embedding Layer**  
  Converts input token indices into dense vector representations. Padding tokens are handled via `padding_idx`.

- **Bidirectional LSTM Encoder**  
  Processes the embedded sequence in both forward and backward directions, capturing context from both sides. The LSTM is set as bidirectional and can be stacked with multiple layers.

- **Dropout Layers**  
  Applied after the embedding and LSTM layers to reduce overfitting and improve generalization.

- **Slot Filling Head**  
  A linear layer (`slot_out`) maps the output of the BiLSTM at each time step to slot label logits, enabling sequence labeling for each token in the utterance.

- **Intent Classification Head**  
  Another linear layer (`intent_out`) takes the concatenated final hidden states from both directions of the BiLSTM and predicts the intent of the entire utterance.

### Forward Pass

1. The input utterance is embedded and passed through dropout.
2. The sequence is packed and processed by the BiLSTM encoder.
3. The output is unpacked and passed through dropout again.
4. For **slot filling**, the model predicts a slot label for each token using the BiLSTM outputs.
5. For **intent classification**, the final hidden states from both LSTM directions are concatenated and passed to the intent classifier.
6. The slot logits are permuted to match the expected shape for loss computation.

## Results
Experiments were done incrementally from the baseline IAS model to the final BiLSTM.
| Model                              | Mean Slot F1 | Mean Intent Acc. |
|-------------------------------------|:------------:|:----------------:|
| Baseline IAS                       |    0.925     |      0.934       |
| IAS + Bidirectionality              |    0.928     |      0.936       |
| IAS + Bidirectionality + Dropout    |    0.940     |      0.941       |
