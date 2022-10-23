import tensorflow as tf
import numpy as np
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._input_size = input_size
        self._step = tf.Variable(1, name = "global_step", trainable = False)
    def build_graph(self):
        
        self._w_1 = tf.Variable(name = "weights_input_hidden",
                                initial_value = tf.random.normal(shape = [self._input_size, self._hidden_size],
                                                                 stddev = 0.1,
                                                                 seed = 2022)
                                )
        
        self._b_1 = tf.Variable(tf.zeros([1, self._hidden_size]),
                                name='biases_input_hidden',
                                )
        
        self._w_2 = tf.Variable(name = "weights_hidden_output",
                                initial_value = tf.random.normal(shape = [self._hidden_size, self._output_size],
                                                                 stddev = 0.1,
                                                                 seed = 2022)
                                )
        
        self._b_2 = tf.Variable(tf.zeros([1, self._output_size]),
                                name='biases_hidden_output',
                                )
        self._trainable_variables = [self._w_1, self._w_2, self._b_1, self._b_2]
    
    def forward(self, X):
        X_tf = tf.cast(X, dtype=tf.float32)
        Z1 = tf.matmul(X_tf, self._w_1) + self._b_1
        Z1 = tf.sigmoid(Z1)
        logits = tf.matmul(Z1, self._w_2) + self._b_2
        return logits
    
    def predict(self, X):
        logits = self.forward(X)
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels

    def compute_loss(self, logits, real_Y):
        
        labels_one_hot = tf.one_hot(indices = real_Y,
                                    depth = self._output_size,
                                    dtype = tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels_one_hot,
                                                       logits = logits)
        loss = tf.reduce_mean(loss)
        return loss
    
    def train_step(self, X, y, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        with tf.GradientTape() as tape:
            logits = self.forward(X)
            current_loss = self.compute_loss(logits, y)
        grads = tape.gradient(current_loss, self._trainable_variables)
        optimizer.apply_gradients(zip(grads, self._trainable_variables))

        return current_loss
    
    def reset_parameters(self):
        self.build_graph()
    
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

def load_datasets():
    with open("../datasets/20news-bydate/words_idfs.txt") as f:
        vocab_size = len(f.read().splitlines())
        
    train_data_reader = DataReader(
        data_path='../datasets/20news-bydate/20news-train-tf-idf.txt',
        batch_size=50,
        vocab_size = vocab_size
    )

    test_data_reader = DataReader(
        data_path='../datasets/20news-bydate/20news-test-tf-idf.txt',
        batch_size=50,
        vocab_size = vocab_size
    )

    return train_data_reader, test_data_reader

def save_parameters(name, value, epoch):
    filename = name.replace(':', '-colon-') + f'-epoch-{epoch}.txt'
    if len(value.shape) == 1:
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number)
                                            for number in value[row]])
                                            for row in range(value.shape[0])])
        
    with open(f'../datasets/20news-bydate/saved-params/{filename}', 'w') as f:
        f.write(string_form)

def restore_parameters(name, epoch):
    filename = name.replace(':', '-colon-') + f'-epoch-{epoch}.txt'
    with open(f'../datasets/20news-bydate/saved-params/{filename}') as f:
        lines = f.read().splitlines()
    if len(lines) == 1:
        value = [[float(number) for number in lines[0].split(',')]]
    else:
        value = [[float(number) for number in lines[row].split(',')]
                for row in range(len(lines))]
    return value


if __name__ == "__main__":
    
    
    train_data_reader, test_data_reader = load_datasets()
    step, MAX_STEP = 0, 1000
    
    #create and train the model
    model = MLP(input_size=len(train_data_reader._data[1]),
                hidden_size=100,
                output_size=20
                )
    model.build_graph()

    while step < MAX_STEP:
        train_data, train_labels = train_data_reader.next_batch()

        loss = model.train_step(X=train_data,
                                y=train_labels,
                                learning_rate=0.01,
                                )
        step += 1
        if step % 100 == 0:
            print(f'Step: {step}, Loss: {loss.numpy()}')
        
        
    #save parameters 
    trainable_variables = model._trainable_variables
    for variable in trainable_variables:
        save_parameters(
            name=variable.name, 
            value=variable.numpy(),
            epoch=train_data_reader._num_epoch,
        )
    
    #reset parameters
    model.build_graph()
    epoch = 1
    #restore parameters
    trainable_variables = model._trainable_variables
    for variable in trainable_variables:
        saved_value = restore_parameters(variable.name,epoch)
        variable.assign(saved_value)
    
    
    #Evaluate on train and test data
    train_data_reader.reset()
    num_true_preds_train = 0
    while True:
        train_data, train_labels = train_data_reader.next_batch()
        train_plabels_eval = model.predict(train_data)
        train_matches = np.equal(train_plabels_eval, train_labels)
        num_true_preds_train += np.sum(train_matches.astype(float))
        if train_data_reader._batch_id == 0:
            break

    print(f'Accuracy on train data: {num_true_preds_train/train_data_reader._data.shape[0]}')
    
    num_true_preds_test = 0
    while True:
        test_data, test_labels = test_data_reader.next_batch()
        test_plabels_eval = model.predict(test_data)
        test_matches = np.equal(test_plabels_eval, test_labels)
        num_true_preds_test += np.sum(test_matches.astype(float))
        if test_data_reader._batch_id == 0:
            break
    print(f'Accuracy on test data: {num_true_preds_test/test_data_reader._data.shape[0]}')
    