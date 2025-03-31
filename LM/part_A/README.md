## Part 1.A: Improving the Baseline ```LM_RNN```

In this part of the project, I applied several modifications to the baseline ```LM_RNN``` model to improve its performance. Each modification was added incrementally, and its impact on Perplexity (PPL) was evaluated. If a modification resulted in worse performance, I removed it and moved on to the next one. 

### Modifications:

1. **Replacing RNN with LSTM**:
   I replaced the baseline RNN model with a Long-Short Term Memory (LSTM) network. This change improved the model's ability to handle long-term dependencies, leading to a reduction in Perplexity.

2. **Adding Dropout Layers**:
   To prevent overfitting, I added two dropout layers:
   - One dropout layer after the embedding layer.
   - One dropout layer before the final linear layer.
   These dropout layers helped to regularize the model, improving its generalization and lowering the Perplexity.

3. **Replacing SGD with AdamW**:
   I replaced the Stochastic Gradient Descent (SGD) optimizer with AdamW. This change led to better training stability and convergence, contributing to a further reduction in Perplexity.

### Hyperparameter Optimization:
I conducted hyperparameter optimization to fine-tune the model. In particular, I experimented with different learning rates, embedding and hidden sizes to minimize Perplexity. The best configuration was found with a learning rate of  $10^{-4}$, used in the third modification, which achieved the lowest Perplexity.

The Perplexity for each experiment is shown below:
- After replacing RNN with LSTM: [--]
- After adding dropout layers: [--]
- After replacing SGD with AdamW: ```111.720```


Full detailed report is available in ```report.pdf```



## Directory Sructure
```
LM/part_A
├── README.md
├── dataset
│   └── PennTreeBank
│       ├── ptb.test.txt
│       ├── ptb.train.txt
│       └── ptb.valid.txt
├── functions.py
├── main.py
├── model.py
├── reports
│   └── test00
│       ├── plot.png
│       ├── ppl_plot.png
│       ├── report.txt
│       └── weights.pt
└── utils.py
```
