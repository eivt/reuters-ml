# reuters-ml

Implementation in Keras of a neural network classifier for the Reuters-21578 dataset, using the `topics` categories as targets and the `lewissplit` split of training and test data. In addition to a simple bag-of-words model, there is also a model which enhances this with a separate channel using an LSTM layer as an encoder over an embedding layer.

Running ```python3 main.py -h``` will return a list of options for running the pipeline.




# Installation

The following will clone the repository and download the reuters-21578 dataset. Note the --recursive option, which will include the BP-MLL submodule.

```
git clone --recursive https://github.com/eivt/reuters-ml.git
cd reuters-ml
wget http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz
mkdir reuters21578
tar xzf reuters21578.tar.gz -C reuters21578
rm reuters21578.tar.gz
```

# Evaluation

None of the classifying models perform particularly well when considering the macro F1 scores. Since this is a multi-label classification problem with a heavily skewed class distribution, attaining a high macro F1 score would require a greater focus on the lesser seen classes. Treating the problem by training C binary classifiers may give better results.

The standard loss function is BP-MLL described in [1], as implemented by [2]. This function appears to increase recall, which makes sense since this loss function is supposed to weight true positives higher than true negatives. However, precision is greatly reduced compared to binary cross-entropy, indicating a high rate of false positives. Depending on the area of usage, we may however still prefer the higher recall rate.

The more complicated LSTM architecture failed to improve upon the basic BoW architecture. It is either a poor architecture choice for the task, or the hyperparameters may be badly chosen; no hyperparameter search has been performed. 

Architecture: BoW feedforward + LSTM over embedding layer
Loss: BP-MLL
Accuracy: 0.0006618133686300463
Micro metrics:
Precision: 0.39704038123902685, Recall: 0.8440415889096241, F1 score: 0.5400426439232409
Macro metrics:
Precision: 0.3039740335648279, Recall: 0.25946206027844326, F1 score: 0.26013507567527516

Architecture: BoW feedforward + LSTM over embedding layer
Loss: binary cross-entropy:
Accuracy: 0.7690271343481139
Micro metrics:
Precision: 0.9325253848673436, Recall: 0.7589976006398294, F1 score: 0.8368606701940036
Macro metrics:
Precision: 0.3756643061828796, Recall: 0.20310875994436195, F1 score: 0.2505208161782147

Architecture: BoW feedforward
Loss: BP-MLL
Accuracy: 0.6892786234281932
Micro metrics:
Precision: 0.7722699386503068, Recall: 0.8389762729938683, F1 score: 0.8042422693585485
Macro metrics:
Precision: 0.3264873487832416, Recall: 0.2668881587799457, F1 score: 0.2773589468229493

Architecture: BoW feedforward
Loss: binary cross-entropy
Accuracy: 0.7729980145598941
Micro metrics:
Precision: 0.9340369393139841, Recall: 0.7549986670221275, F1 score: 0.8350287483414418
Macro metrics:
Precision: 0.38598244424038813, Recall: 0.20573373108414617, F1 score: 0.25384756009080667

# References
[1] Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with applications to functional genomics and text categorization." IEEE transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.

[2] Huwald, Lukas. https://github.com/vanHavel/bp-mll-tensorflow