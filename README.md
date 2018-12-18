# reuters-ml

Implementation in Keras of a neural network classifier for the Reuters-21578 dataset, using the `topics` categories as targets and the `lewissplit` split of training and test data. In addition to a simple bag-of-words model, there is also a model which enhances this with a separate channel using an LSTM layer as an encoder over an embedding layer.

Running ```python3 main.py -h``` will return a list of options for running the pipeline.

The standard loss function used is BP-MLL by [1], as implemented by [2].

Neither of the two classifying models perform particularly well. This is a multi-label classification problem with a heavily skewed class distribution. A higher macro F1 score may be attainable using a loss function that somehow prioritises the less populated classes. Treating the problem by training C binary classifiers is also likely to give better results.

# References
[1] Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with applications to functional genomics and text categorization." IEEE transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.
[2] Huwald, Lukas. https://github.com/vanHavel/bp-mll-tensorflow