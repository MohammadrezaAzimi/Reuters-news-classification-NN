# Reuters-news-classification-NN


# Project description:
In this project, there are 8982 news that belong to 46 mutually exclusive categories. Training and development (hold-out) sets are constructed from theses examples. The first 1000 news are kept as the development set while the rest is used for training set. There is an additional test set of 2246 news. 

# Network model: 
A three-layer neural network is used. The network is comprised of two fully-connected layers with 64 hidden units and rectified linear unit (ReLU) activation as well as the output layer with softmax activation that classifies the news as one of the 46 classes.

# Representing the class labels as one hot vectors 
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Details: 
Optimization of stochastic gradient descent algorithm is done using RMSProp. Accuracy is performance metric and categorical_crossentropy is the loss function.

Here are the tips for implementation:

1- Only 10000 most frequently words in the training set are kept.
2- Using Python code reuters.get_word_index() words are mapped to integer indices. Then a dictionary is constructed using dict().
3- To prepare the data, integer indices are transformed to tensors using one-hot encoding. In Keras, the built-in function to_categorical() is defined for this purpose.
4- The batch size set equal to 512 and then fitted the training data to the model over 20 epochs.

# Modification:
To combat the overfitting problem, early stoppage is used and the training is done only for 9 epochs. The resulting accuracy is 0.8.
