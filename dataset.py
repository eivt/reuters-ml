from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
import logging
import keras
import os
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()


def parse_sgm_files(data_home, splits: str):
    '''
    Parse the data files and put the data into an easily managable data structure
    :param data_home: path to the directory containing the Reuters data files
    :return: Pandas DataFrame containing the
    '''

    # create a mapping from topics to topic identifiers
    with open(os.path.join(data_home, 'all-topics-strings.lc.txt')) as f:
        topics_raw = re.findall(r'[^\s]+', f.read())
    topic_mapping = dict(zip(topics_raw, range(len(topics_raw))))

    file_paths = [os.path.join(data_home, 'reut2-0{:02d}.sgm'.format(i)) for i
                  in range(22)]

    documents = []
    for path in file_paths:
        logger.info('Reading {}'.format(path))

        raw = open(path, 'r', errors='ignore').read()
        tree = BeautifulSoup(raw, 'html.parser')

        for document in tree.find_all('reuters'):

            topics = [topic_mapping[t.text] for t in document.topics.children]
            if len(topics) == 0:
                # skip documents without a topic
                continue

            text = document.find('text').text

            split = document.attrs[splits]

            documents.append({'topics': np.array(topics),
                              'split': split,
                              'text': text})

    df = pd.DataFrame.from_dict(documents)

    return df, topic_mapping


class Dataset:
    def __init__(self, data_home, vocab_size=14151, dev_set=True,
                 splits='lewissplit'):
        '''
        Loads the data into memory and trains a tokenizer on the training data
        :param data_home: path to the directory containing the Reuters data files
        :param vocab_size: the number of words to include in the vocabulary
        '''
        self.df, self.topic_mapping = parse_sgm_files(data_home, splits)
        self.n_classes = len(self.topic_mapping)
        self.vocab_size = vocab_size
        self.dev_set = dev_set

        # Extract the sample indices corresponding to the different splits
        self.train_ids = list(self.df.index[self.df['split'] == 'TRAIN'])
        self.test_ids = list(self.df.index[self.df['split'] == 'TEST'])

        # If we have requested a dev set, we take a tenth out from the training set
        if dev_set:
            self.dev_ids = random.sample(self.train_ids,
                                         int(len(self.train_ids) / 10))
            for id in self.dev_ids:
                self.train_ids.remove(id)


        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=vocab_size,
            oov_token='$OOV')

        assert self.vocab_size == self.tokenizer.num_words
        self.tokenizer.fit_on_texts(self.df.loc[self.train_ids]['text'])

    def bow_data_batch(self, batch_ids, mode='tfidf'):
        '''
        Generates a bag-of-words representation for the requested texts
        :param batch_ids: indices of the samples to generate representations of
        :param mode: which vectorization mode to pass on to the tokenizer
        :return: 2d numpy array shape (batch size, vocabulary size)
        '''
        batch_content = self.df.loc[batch_ids]['text']
        content_matrix = self.tokenizer.texts_to_matrix(batch_content,
                                                        mode=mode)
        return content_matrix

    def vocab_sequence_batch(self, batch_ids):
        '''
        Generates sequences of vocabulary identifiers for the requested texts
        :param batch_ids: indices of the samples to generate representations of
        :return: 2d numpy array, shape (batch size, longest sequence in batch)
        '''
        batch_content = self.df.loc[batch_ids]['text']
        content_matrix = self.tokenizer.texts_to_sequences(batch_content)

        # the sequences are padded to the length of the longest document in the batch
        sequence_length = np.max([len(text) for text in content_matrix])
        content_matrix = keras.preprocessing.sequence.pad_sequences(
            content_matrix, maxlen=sequence_length)
        return content_matrix

    def bow_vocab_batch(self, batch_ids):
        '''
        Generates both vocabulary id sequences and bag-of-words representations
        for the requested texts
        :param batch_ids: indices of the samples to generate representations of
        :return: list of two 2d numpy arrays
        '''
        bow = self.bow_data_batch(batch_ids)
        sequence = self.vocab_sequence_batch(batch_ids)
        return [bow, sequence]

    def target_batch(self, batch_ids):
        '''
        Generates multi-hot target representations for the requested texts
        :param batch_ids: indices of the samples to generate target vectors for
        :return: 2d numpy array containing the target vectors
        '''
        target_vectors = np.zeros((len(batch_ids), self.n_classes),
                                  dtype=int)

        for i, topics in enumerate(self.df.loc[batch_ids]['topics']):
            for topic_id in topics:
                target_vectors[i][topic_id] = 1
        return target_vectors

    def batch_generator(self, data_set, data_type, batch_size=32):
        '''
        Creates a Sequence which generates training batches
        :param data_set: which of either the train or test set to generate representations for
        :param data_type: what type of representation to generate
        :param batch_size: number of samples in each batch
        :return: Sequence for the specified set of indices
        '''

        # determine the dataset indices given the requested set
        if data_set == 'train':
            ids = self.train_ids
        elif data_set == 'test':
            ids = self.test_ids
        elif data_set == 'dev':
            ids = self.dev_ids
        else:
            raise ValueError('Invalid set: {}'.format(data_set))

        # determine the function to use to format the data
        if data_type == 'bow':
            data_vectorizer = self.bow_data_batch
        elif data_type == 'vocab_seq':
            data_vectorizer = self.vocab_sequence_batch
        elif data_type == 'bow_vocab':
            data_vectorizer = self.bow_vocab_batch

        else:
            raise ValueError('Invalid data format type: {}'.format(data_type))

        return Batch_Sequence(ids, batch_size, data_vectorizer,
                              self.target_batch)


class Batch_Sequence(keras.utils.Sequence):
    def __init__(self, ids, batch_size, data_vectorizer, target_vectorizer):
        '''
        A safe structure for generating training data from raw text
        :param ids: indices to the samples to generate
        :param batch_size: number of samples in each batch
        :param data_vectorizer: method used to formatting the sample data
        :param target_vectorizer: method to format the target vectors
        '''
        self.data_vectorizer = data_vectorizer
        self.target_vectorizer = target_vectorizer
        self.ids = ids
        np.random.shuffle(self.ids)
        self.batch_size = batch_size
        self.len = int(np.ceil(len(ids) / batch_size))

    def __len__(self):
        '''
        Number of batches in the Sequence
        :return: The number of batches in the Sequence
        '''
        return self.len

    def __getitem__(self, index):
        '''
        Gets batch at position `index'.
        :param index: position of the batch in the Sequence
        :return: A batch
        '''
        start_i = index * self.batch_size
        end_i = (index + 1) * self.batch_size
        batch_ids = self.ids[start_i:end_i]  # list slicing fails gracefully

        content_matrix = self.data_vectorizer(batch_ids)
        target_matrix = self.target_vectorizer(batch_ids)

        return (content_matrix, target_matrix)

    def on_epoch_end(self):
        '''Shuffles the data between each iteration'''
        np.random.shuffle(self.ids)
