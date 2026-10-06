# GPT-2 From Scratch

Implementing GPT-2 from scratch following Build Large Language Model from Scratch [1] by Sebastien Raschka. No coding assisstant has been used during the exercice: it's more instructive and way more fun done this way !

### Fine-Tuning for Classification: Alternative Dataset and Methodology

[WIP]

For this section I deliberately choose to diverge from the book: I will use a different dataset/task, one
that interests me more than spam/ham classification, namely ICLR paper status.
I will train the classifier to predict the status given the paper's title.
The dataset I'll be using is on Hugging Face: https://huggingface.co/datasets/Vidushee/iclr-papers-with-code-1k, and I'll be using
Hugging Face's `datasets` package instead of pandas.

The dataset is highly unbalanced, but I keep the original class distribution rather than undersampling the majority "rejected" class, preserving all available training examples (the dataset has 944 "rejected" instances and only "107" accepted ones). Class imbalance is partially addressed at the decision stage by tuning the classification threshold on the validation set although I still recognize that threshold tuning does not eliminate its potential effect on model training.

## References

[1] Raschka, S. (2024). *Build a Large Language Model (From Scratch)*. Manning Publications.