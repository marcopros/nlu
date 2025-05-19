## Part 2.B : JointBERT for Slot Filling and Intent Classification

### Model Architecture
-----------------------

This project implements a **JointBERT-based model** for slot filling and intent classification on the ATIS dataset.

#### Components

- **BERT Encoder**  
  The core of the model is a pre-trained BERT encoder (`BertModel`), which produces contextualized embeddings for each token in the input utterance.

- **Dropout Layer**  
  Dropout is applied to both the token-level and sentence-level representations to reduce overfitting.

- **Intent Classification Head**  
  A linear layer (`intent_classifier`) takes the pooled output corresponding to the `[CLS]` token and predicts the intent of the entire utterance.

- **Slot Filling Head**  
  Another linear layer (`slot_classifier`) takes the contextualized embeddings of all tokens and predicts a slot label for each token (sequence labeling).

#### Forward Pass

1. The input utterance is tokenized and passed through the BERT encoder.
2. The model extracts:
   - The last hidden states (token-level representations) for slot filling.
   - The pooled `[CLS]` output (sentence-level representation) for intent classification.
3. Dropout is applied to both representations.
4. The intent classifier predicts the intent from the `[CLS]` embedding.
5. The slot classifier predicts slot labels for each token from the sequence embeddings.
6. The loss is computed as the sum of:
   - Cross-entropy loss for intent classification.
   - Cross-entropy loss for slot filling, computed only on valid tokens (ignoring sub-tokens and special tokens using the `-100` ignore index).


## Results
Experiments were done using both `bert-base-uncased` and `bert-large-uncased` pretrained models.
| Model                              | Mean Slot F1 | Mean Intent Acc. |
|-------------------------------------|:------------:|:----------------:|
| JointBERT  (base)                    |    0.951     |      0.969       |
| JointBERT (large)              |    0.955     |      0.978       |