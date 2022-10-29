import numpy as np

class DataReader:
    def __init__ (self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        self._batch_id = 0
        self._num_epoch = 0
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        
        self._data = np.empty((len(d_lines), vocab_size))
        self._labels = np.empty(len(d_lines))
        for data_id, line in enumerate(d_lines):
            r_d = np.zeros(vocab_size)
            features = line.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index, value = int(token.split(":")[0]), float(token.split(":")[1])
                r_d[index] = value
            self._data[data_id] = r_d
            self._labels[data_id] = label
    
    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1
        
        if end + self._batch_size  > self._data.shape[0]:
            end = self._data.shape[0]
            self._num_epoch += 1
            self._batch_id = 0
            indices = np.arange(self._data.shape[0])
            np.random.seed(2022)
            np.random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]
            
        return self._data[start:end], self._labels[start:end]
    def reset(self):
        self._batch_id = 0
        self._num_epoch = 0
