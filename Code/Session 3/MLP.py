import tensorflow as tf
import numpy as np
from load_and_save import load_datasets, save_parameters, restore_parameters
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
    