from DataReader import DataReader
from RNN import RNN
import tensorflow.compat.v1 as tf
import numpy as np

def train_and_evaluate_RNN():
    with open('../datasets/w2v/vocab-raw.txt') as f:
        vocab_size = len(f.read().splitlines())
    tf.set_random_seed(2022)
    rnn = RNN(vocab_size = vocab_size,
                embedding_size = 300,
                lstm_size = 50,
                batch_size = 50)
    predicted_labels, loss = rnn.build_graph()
    train_op = rnn.trainer(loss = loss, learning_rate = 0.01)
    with tf.Session() as sess:
        train_data_reader = DataReader(data_path = '../datasets/w2v/20news-train-encoded.txt',
                                        batch_size = 50)
        test_data_reader = DataReader(data_path = '../datasets/w2v/20news-test-encoded.txt',
                                        batch_size = 50)
        step = 0
        MAX_STEP = 1000
        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            next_train_batch = train_data_reader.next_batch()
            train_data, train_labels, train_sentence_lengths, trainfinal_tokens = next_train_batch
            plabels_eval, loss_eval, _ = sess.run([predicted_labels, loss, train_op],
                                                    feed_dict = {rnn._data: train_data,
                                                                rnn._labels: train_labels,
                                                                rnn._sentence_lengths: train_sentence_lengths,
                                                                rnn.final_tokens: trainfinal_tokens})
            step += 1
            if step % 20 == 0:
                print(f'loss: {loss_eval}')
            if train_data_reader.batch_id == 0:
                num_true_preds = 0
                while True:
                    next_test_batch = test_data_reader.next_batch()
                    test_data, test_labels, test_sentence_lengths, testfinal_tokens = next_test_batch
                    test_plabels_eval = sess.run(predicted_labels, feed_dict = {rnn._data: test_data,
                                                                                rnn._labels: test_labels,
                                                                                rnn._sentence_lengths: test_sentence_lengths,
                                                                                rnn.final_tokens: testfinal_tokens})
                    matches = np.equal(test_plabels_eval, test_labels)
                    num_true_preds += np.sum(matches.astype(float))
                    if test_data_reader.batch_id == 0:
                        break
                print(f'Epoch: {train_data_reader.num_epoch}')
                print(f'Accuracy on test data: {num_true_preds * 100 / test_data_reader._size}')
        
if __name__ == '__main__':
    train_and_evaluate_RNN()