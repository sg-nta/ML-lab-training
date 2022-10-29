
import numpy as np
from constants import *

class DataReader:
    def __init__(self, data_path, batch_size):
        self._batch_size = batch_size
        self.num_epoch = 0
        self.batch_id = 0
        with open(data_path) as f:
            doc_size = sum(1 for _ in f)

        self.data = np.zeros((doc_size, MAX_SENTENCE_LENGTH), dtype=np.int32)
        self.labels = np.empty(doc_size)
        self.sentence_lengths = np.empty(doc_size)
        self.final_tokens = []
        with open(data_path) as f:
            for data_id, line in enumerate(f):
                features = line.split('<fff>')
                label = int(features[0])
                sentence_length = int(features[2])
                tokens = [int(i) for i in features[3].split()]

                self.data[data_id] = tokens
                self.labels[data_id] = label
                self.sentence_lengths[data_id] = sentence_length
                self.final_tokens.append(tokens[-1])
        self.final_tokens = np.array(self.final_tokens)
        self._size = len(self.data)
        
    def next_batch(self):
        start = self.batch_id * self._batch_size
        end = start + self._batch_size
        self.batch_id += 1

        if end + self._batch_size > self.data.shape[0]:
            self._size = end
            end = self.data.shape[0]
            start = end - self._batch_size
            self.num_epoch += 1
            self.batch_id = 0
            indices = np.arange(self.data.shape[0])
            np.random.seed(42)
            np.random.shuffle(indices)
            self.data, self.labels, self.sentence_lengths, self.final_tokens = \
                self.data[indices], self.labels[indices], self.sentence_lengths[indices], self.final_tokens[indices]

        return self.data[start:end], self.labels[start:end], self.sentence_lengths[start:end], self.final_tokens[start:end]
