import argparse
import logging
import keras
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from sklearn import metrics
from bpmll.bp_mll_keras import bp_mll_loss
from dataset import Dataset

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()


def mlp_bow_model(vocab_size, n_classes, layers):
    '''
    Creates a multilayer perceptron with one hidden layer and bags of words as input
    :param vocab_size: number of words in the input vocabulary
    :param n_classes: number of output classes
    :return: the untrained model
    '''
    # add input layer
    input = Input(shape=(vocab_size,))

    # add hidden layers
    prev_layer = input
    for layer_units in layers:
        hidden = Dense(layer_units, activation='relu')(prev_layer)
        prev_layer = hidden

    # the output layer will use sigmoid since this is a multi-label problem
    output = Dense(n_classes, activation='sigmoid')(prev_layer)

    model = keras.models.Model(input, output)
    return model


def lstm_model(vocab_size, n_classes, emb_dims=300, rec_units=300, dropout=0.0,
               hidden_units=300):
    '''
    Creates a model that combines a one-layer multilayer perceptron with an LSTM
    on top of an embedding layer. This does not actually perform very well.
    :param vocab_size: number of words in the input vocabulary
    :param n_classes: number of output classes
    :param emb_dims: output dimension of the embedding layer
    :param rec_units: recurrent units in the LSTM layer
    :param dropout: dropout for the recurrent layer
    :param hidden_units: number of units in the hidden layer over the BoW input
    :return: the untrained model
    '''
    sequence_input = Input(batch_shape=(None, None), name='seq_input')
    embedding = Embedding(input_dim=vocab_size + 1,
                          input_length=None,
                          output_dim=emb_dims,
                          mask_zero=True,
                          )(sequence_input)
    lstm = LSTM(units=rec_units,
                recurrent_dropout=dropout)(embedding)

    bow_input = Input(shape=(vocab_size,), name='bow_input')
    hidden = Dense(units=hidden_units, activation='relu')(bow_input)

    concat = Concatenate()([lstm, hidden])

    output = Dense(n_classes, activation='sigmoid')(concat)

    model = keras.models.Model([bow_input, sequence_input], output)
    return model


def compile_train(model, dataset, data_type, loss, epochs=10):
    '''
    Compiles and trains the model
    :param model: the Keras model to train
    :param dataset: dataset containing training
    :param loss: loss function to use, 'bpmll' or 'binary_crossentropy'
    :return: the trained model
    '''

    if loss == 'bpmll':
        loss = bp_mll_loss

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    model.summary()

    # if a dev set has been set aside, we will use it for early stopping
    if dataset.dev_set:
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                       min_delta=0.001,
                                                       patience=2,
                                                       verbose=2)
        model.fit_generator(
            generator=dataset.batch_generator(data_type=data_type,
                                              data_set='train'),
            validation_data=dataset.batch_generator(data_type=data_type,
                                                    data_set='dev'),
            callbacks=[early_stopping],
            epochs=epochs,
            verbose=2)
    else:
        model.fit_generator(
            generator=dataset.batch_generator(data_type=data_type,
                                              data_set='train'),
            epochs=epochs,
            verbose=2)
    return model


def evaluate(model, dataset, data_type):
    '''
    Evaluates the model on relevant metrics, and prints the result to screen
    :param model: a trained model
    :param dataset: dataset containing samples
    '''

    # get the true targets and our predictions
    test_generator = dataset.batch_generator(data_type=data_type,
                                             data_set='test')

    targets = dataset.target_batch(test_generator.ids)
    predict_probs = model.predict_generator(test_generator)
    predictions = np.where(predict_probs > 0.5, 1, 0)

    # produce our evaluation metrics
    accuracy = metrics.accuracy_score(targets, predictions)
    micro_precision, micro_recall, micro_fscore, _ = metrics.precision_recall_fscore_support(
        targets, predictions, average='micro')
    macro_precision, macro_recall, macro_fscore, _ = metrics.precision_recall_fscore_support(
        targets, predictions, average='macro')

    print('Accuracy: {}'.format(accuracy))
    print('Micro metrics:\nPrecision: {}, Recall: {}, F1 score: {}'.format(
        micro_precision, micro_recall, micro_fscore))
    print('Macro metrics:\nPrecision: {}, Recall: {}, F1 score: {}'.format(
        macro_precision, macro_recall, macro_fscore))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('reuters',
                        help='path to directory containing the reuters21578 dataset')
    parser.add_argument('--earlystop',
                        help='set aside a dev set and use early stopping',
                        action='store_true')
    parser.add_argument('--lstm',
                        help='train embeddings and an lstm encoder model alongside the bag-of-words model',
                        action='store_true')
    parser.add_argument('--loss',
                        help='which loss function to use',
                        choices=['bpmll', 'binary_crossentropy'],
                        default='bpmll')
    args = parser.parse_args()

    dataset = Dataset(args.reuters, dev_set=args.earlystop)

    logger.info('Defining model architecture')
    if args.lstm:
        data_type = 'bow_vocab'
        model = lstm_model(dataset.vocab_size,
                           dataset.n_classes,
                           emb_dims=300,
                           rec_units=300,
                           dropout=0.1)
    else:
        data_type = 'bow'
        model = mlp_bow_model(dataset.vocab_size,
                              dataset.n_classes,
                              layers=[300])

    logger.info('Training model')
    model = compile_train(model, dataset, data_type, loss=args.loss)

    logger.info('Evaluating model')
    evaluate(model, dataset, data_type)
