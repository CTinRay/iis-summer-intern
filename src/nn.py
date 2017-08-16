import tensorflow as tf
from tqdm import tqdm
from utils import BatchGenerator, make_dir
import os
import pdb


class NNClassifier:
    def _inference(self, X):
        hidden_size = 200
        word_embedding = tf.constant(self._embedding)
        #(N_example , N_words, embed_size)
        embedding_b = tf.nn.embedding_lookup(word_embedding, X[:, 0, :])
        embedding_q = tf.nn.embedding_lookup(word_embedding, X[:, 1, :])
        # Encode Body into a LSTM output cell
        with tf.variable_scope('Body_GRU'):
            cell_b = tf.contrib.rnn.GRUCell(hidden_size)
            output_b, out_state_b = tf.nn.dynamic_rnn(cell_b, embedding_b, dtype = tf.float32)
        with tf.variable_scope('Question_GRU'):
            cell_q = tf.contrib.rnn.GRUCell(hidden_size)
            output_q, out_state_q = tf.nn.dynamic_rnn(cell_q, embedding_q, dtype = tf.float32)

        # output is [batch_size, N_words(timestep), hidden_size], 
        # we need last timestep output : (batch_size, hidden_size)
        output_b = output_b[:, -1, :]
        output_q = output_q[:, -1, :]
        # output is [batch_size, 2*hidden_size]
        output = tf.concat([output_b, output_q], axis = 1)
        # (batch_size, _n_classes)
        dense = tf.layers.dense(inputs=output,
                                units=self._n_classes)
        return dense

    def _iter(self, X, y, tensor_loss, train_op, placeholder, metric_tensors):
        # initialize local variable for metrics
        self._session.run(tf.local_variables_initializer())

        # make accumulator for metric scores
        metric_scores = {}
        for metric in self._metrics:
            metric_scores[metric] = 0

        # make generator
        batch_generator = BatchGenerator(X, y, self._batch_size)

        # run batches for train
        for b in tqdm(range(X.shape[0] // self._batch_size)):
            batch = next(batch_generator)
            feed_dict = {placeholder['x']: batch['x'],
                         placeholder['y']: batch['y']}

            if train_op is not None:
                loss, _, metrics \
                  = self._session.run([tensor_loss, train_op, metric_tensors],
                                      feed_dict=feed_dict)
            else:
                loss, metrics \
                  = self._session.run([tensor_loss, metric_tensors],
                                      feed_dict=feed_dict)

        # put metric score in summary and print them
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=float(loss))
        print('loss=%f' % loss)
        for metric in self._metrics:
            score = float(metrics[metric][0])
            summary.value.add(tag=metric,
                              simple_value=score)
            print(', %s=%f' % (metric, score), end='')

        print('\n', end='')
        return summary

    def __init__(self, learning_rate=1e-1, batch_size=1,
                 n_iters=10, name='dnn', valid=None, embedding=None):
        self._batch_size = batch_size
        self._n_iters = n_iters
        self._metrics = {'accuracy': tf.metrics.accuracy}
        self._loss = tf.losses.softmax_cross_entropy
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._name = name
        self._valid = valid
        self._embedding = embedding

    def fit(self, X, y):

        self._n_classes = y.shape[1]
        # make directories to store training process and models
        make_dir(self._name)
        make_dir(os.path.join(self._name, 'train'))
        make_dir(os.path.join(self._name, 'valid'))

        # input placeholders
        placeholder = {'x': tf.placeholder(tf.int32,
                                           shape=(None, X.shape[1], X.shape[2])),
                       'y': tf.placeholder(tf.int32, shape=(None, y.shape[1]))}

        # connect nn
        with tf.variable_scope('nn') as scope:
            y_prob = self._inference(placeholder['x'])
            loss = self._loss(placeholder['y'], y_prob)
            train_op = self._optimizer.minimize(loss)

            # make metric tensors
            metric_tensors = {}
            for metric in self._metrics:
                y_max = tf.reduce_max(y_prob, axis=-1)
                y_pred = tf.cast(tf.equal(y_prob, tf.reshape(y_max, (-1, 1))),
                                 dtype=tf.int32)
                metric_tensors[metric] = \
                    self._metrics[metric](placeholder['y'], y_pred)

        self._session = tf.Session()
        # prepare summary writer
        summary_writer = \
            {'train': tf.summary.FileWriter(
                os.path.join(self._name, 'train'), self._session.graph),
             'valid': tf.summary.FileWriter(
                 os.path.join(self._name, 'valid'), self._session.graph)}

        # initialization
        self._session.run(tf.global_variables_initializer())

        # Start the training loop.
        for i in range(self._n_iters):
            # train and evaluate train score
            print('training %i' % i)
            summary = self._iter(X, y, loss, train_op,
                                 placeholder, metric_tensors)
            summary_writer['train'].add_summary(summary, i)
            summary_writer['train'].flush()

            # evaluate valid score
            if self._valid is not None:
                print('evaluating %i' % i)
                summary = self._iter(self._valid['x'],
                                     self._valid['y'], loss, None,
                                     placeholder, metric_tensors)
                summary_writer['valid'].add_summary(summary, i)
                summary_writer['valid'].flush()

    def predict(self, X):
        with tf.variable_scope('nn', reuse=True):
            X_placeholder = tf.placeholder(
                tf.int32, shape=(None, X.shape[1], X.shape[2]))
            y_prob = self._inference(X_placeholder)
            y_max = tf.reduce_max(y_prob, axis=-1)
            y_pred = tf.cast(tf.equal(y_prob, tf.reshape(y_max, (-1, 1))), dtype=tf.int32)

            y_ = self._session.run(y_pred,
                                   feed_dict={X_placeholder: X})

        return y_
