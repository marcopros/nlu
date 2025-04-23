from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch

class JointBERT(BertPreTrainedModel):
    def __init__(self, config, num_intents, num_slots):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slots)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, slot_labels = None, intent_label = None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)
        pooled_output = self.dropout(outputs.pooler_output)

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = 0
        loss_fct = nn.CrossEntropyLoss()
        if intent_label is not None:
            loss += loss_fct(intent_logits, intent_label)
        if slot_labels is not None:
            # Flatten the tokens for slot loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = slot_logits.view(-1, slot_logits.shape[-1])[active_loss]
            active_labels = slot_labels.view(-1)[active_loss]
            loss += loss_fct(active_logits, active_labels)
        return loss, intent_logits, slot_logits