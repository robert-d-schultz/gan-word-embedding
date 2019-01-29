Code related to *Generative Adversarial Networks and Word Embeddings for Natural Language Generation*, my master's thesis.

## Requirements
PyTorch version 0.4.1
tensorboardX for data visualization
Some type of word embeddings (code is set up to use 50d GloVe https://nlp.stanford.edu/projects/glove/)
gensim for word embedding manipulation
Training dataset (formatted one sentence per line)

## Instructions:
Place the word embedding file in the "raw" folder
Place the training data in the "raw" folder
Run preprocessing.py
Run main.py to train a model
The model will be saved to the "models" folder
Run main.py --generate to generate sentences
