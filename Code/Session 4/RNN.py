from collections import defaultdict
from genericpath import isfile
from os import listdir
import re
import tensorflow as tf
import numpy as np

MAX_DOC_LENGTH = 500
NUM_CLASSES = 10
def gen_data_and_vocab():
    def collect_data_from(parent_path, newsgroup_list, word_count = None):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + '/' + newsgroup + '/'
            
            files = [(filename, dir_path + filename) 
                     for filename in listdir(dir_path)
                     if isfile(dir_path + filename)]
            files.sort()
            label = group_id
            print(f'Processing: {group_id}-{newsgroup}')
            for filename, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()
                    words = re.split('\W+', text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data
    word_count = defaultdict(int)
    path = '../datasets/20news-bydate/'
    parts = [path + dir_name + '/' 
                 for dir_name in listdir(path)
                 if not isfile(path + dir_name)]
    
    train_path, test_path = (parts[0], parts[1]) if 'train' in parts[0] else (parts[1], parts[0])
    newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]
    newsgroup_list.sort()
    
    train_data = collect_data_from(
        parent_path=train_path,
        newsgroup_list=newsgroup_list,
        word_count=word_count
    )
    
    vocab = [word for word, freq in zip(word_count.keys(), word_count.values()) if freq > 10]
    vocab.sort()
    with open('../datasets/w2v/vocab-raw.txt','w') as f:
        f.write('\n'.join(vocab))
        
    test_data = collect_data_from(
        parent_path=test_path,
        newsgroup_list=newsgroup_list
    )
    
    with open('../datasets/w2v/20news-train-raw.txt','w') as f:
        f.write('\n'.join(train_data))
    with open('../datasets/w2v/20news-test-raw.txt','w') as f:
        f.write('\n'.join(test_data))
    

def encode_data(data_path, vocab_path):
    unknown_ID = 1
    padding_ID = 0
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2)
                      for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])
                     for line in f.read().splitlines()]
    encoded_data = []
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)
        
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_ID))
        
        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(words)
            for _ in range(num_padding):
                encoded_text.append(str(padding_ID))
        encoded_data.append(f'{label}<fff>{doc_id}<fff>{sentence_length}<fff>' + ' '.join(encoded_text))
        
    dir_name = '/'.join(data_path.split('/')[:-1])
    filename = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(dir_name + '/' + filename, 'w') as f:
        f.write('\n'.join(encoded_data))

class RNN:
    def __init__(self,vocab_size,embedding_size, lstm_size, pretrained_w2v_path, batch_size):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._lstm_size = lstm_size
        self._batch_size = batch_size
        self._data = tf.Variable(tf.zeros(shape = [batch_size, MAX_DOC_LENGTH]), dtypes = tf.int32)
        self._labels = tf.Variable(tf.zeros(shape = [batch_size,]), dtypes = tf.int32)
        self._sentence_lengths = tf.Variable(tf.zeros(shape = [batch_size,]), dtypes = tf.int32)
        self._final_tokens = tf.Variable(tf.zeros(shape = [batch_size,], dtype = tf.int32))
    def embedding_layer(self,indices):
        pretrained_vectors = []
        pretrained_vectors.append(np.zeros(self._embedding_size))
        np.random.seed(2022)
        for _ in range(self._vocab_size + 1):
            pretrained_vectors.append(np.random.normal(loc=0,
                                                       scale=1.,
                                                       size=self._embedding_size))
        pretrained_vectors = np.array(pretrained_vectors)
        
        self._embedding_matrix = tf.Variable(name='embedding',
                                             initial_value=tf.fill([self._vocab_size + 2, self._embedding_size],
                                                                   pretrained_vectors))
        
        return tf.nn.embedding_lookup(self._embedding_matrix, indices)
    
    def LSTM_layer(self, embeddings):
        lstm_cell = tf.keras.layers.LSTMCell(self._lstm_size)
        zero_state = tf.zeros(shape=(self._batch_size, self._lstm_size))
        
        return
    
    def build_graph(self):
        embeddings = self.embedding_layer(self._data)
        lstm_outputs = self.LSTM_layer(embeddings)
        
        weights = tf.Variable(name = 'final_layer_weights',
                              initial_value = tf.random.normal(shape = [self._lstm_size, NUM_CLASSES],
                                                               stddev = 0.1,
                                                               seed = 2022))
        biases = tf.Variable(name = 'final_layer_biases',
                             initial_value = tf.zeros([1, NUM_CLASSES])
                            )
        logits = tf.matmul(lstm_outputs, weights) + biases
        labels_one_hot = tf.one_hot(indices = self._labels,
                                    depth = NUM_CLASSES,
                                    dtype=tf.float32)
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels_one_hot,
                                                       logits=logits)
        
        loss = tf.reduce_mean(loss)
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis = 1)
        predicted_labels = tf.squeeze(predicted_labels)
        
    def trainer(self, loss, learning_rate):
        return
    
if __name__ == '__main__':
    encode_data('../datasets/w2v/20news-test-raw.txt', '../datasets/w2v/vocab-raw.txt')
    encode_data('../datasets/w2v/20news-train-raw.txt', '../datasets/w2v/vocab-raw.txt')
    