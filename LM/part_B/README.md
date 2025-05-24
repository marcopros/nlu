## Part 1.B: Advanced Regularisation Techniques
For the part 1.B of the project, I've implemented some advanced techniques to the Language Model, such as Weight Tying, Variational Dropout and the non-monotonically triggered AvSGD.
### Weight Tying
Weight tying is a technique used in neural networks to reduce the number of parameters and improve generalization. In the context of language models, it involves sharing weights between the input and output embeddings. This means that the same set of weights is used for both the input layer (to convert words into vectors) and the output layer (to convert vectors back into word probabilities). By tying these weights, we can reduce the number of parameters in the model, which can help prevent overfitting and improve performance on unseen data.

### Variational Dropout
Variational dropout is a technique used to improve the robustness of neural networks by applying dropout in a more structured way. In standard dropout, a new binary mask is sampled every time the dropout funcion is called(e. g., independently for each timestep in an LSTM). On the other hand, Variational Dropout (Gal et Ghahramani, 2016) is sampled once at the first timestep and locked for all repeated connections (shared weights) in both forward and backward passes. This preserves consistency across timesteps, a critical issue for RNNs/LSTMs to relate long-term dependencies.

### AvSGD

Stochastic Gradient Descent (SDG) is a cornerstone for training DL models, offering theoretical guarantees like linear convergence and saddle-point avoidance. In neural language modelling, vanilla SGD often outperforms adaptive methods (e.g., Adam, RMSProp). However, Averaged SGD (AvSGD)—which returns the mean of iterates after a trigger point 
T - has seen limited adoption due to tuning challenges for 
T and learning-rate schedules.
I've implemented the **Non-Monotonically Triggered Averaged SGD** in this way:
1. **Triggering Mechanism**: The code switches from SGD to AvSGD when the validation loss fails to improve for ```INTERVAL = 3``` consecutive checks (non-monotonic criterion).
2. **Averaging Implementation**: Uses PyTorch’s ```optim.ASGD```, which averages weights after triggering (```t0``` marks the start of averaging).
During evaluation, it temporarily swaps model weights with the averaged weights (```optimizer.state[prm]['ax']```) to compute validation perplexity, then reverts to the latest weights for training.
3. **Learning Rate**: The learning rate (```lr=2```) remains fixed until a StepLR scheduler decays it (every 5 epochs, ```gamma=0.75```).
