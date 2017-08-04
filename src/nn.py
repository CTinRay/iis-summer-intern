import tensorflow as tf
from tqdm import tqdm
from utils import BatchGenerator
import os


class NNClassifier:
    def _inference(self, X):
        # [TODO] wrote model here
        dense = tf.layers.dense(inputs=X[:, 0, :],
                                units=self._n_classes,
                                activation=tf.nn.relu)
        return dense

    def _train(self, loss):
        optimizer = self._optimizer
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def _iter(self, session, X, y, train_op, placeholder, metric_tensors):
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
                _, batch_scores = session.run([train_op, metric_tensors],
                                              feed_dict=feed_dict)
            else:
                batch_scores = session.run(metric_tensors,
                                           feed_dict=feed_dict)

            # accumulate metric score of batch
            # and average them by weight of batch size
            score_weight = batch['x'].shape[0] / X.shape[0]
            for metric in self._metrics:
                metric_scores[metric] += \
                    batch_scores[metric] * score_weight

        # put metric score in summary and print them
        summary = tf.Summary()
        for metric in self._metrics:
            summary.value.add(tag=metric,
                              simple_value=metric_scores[metric])
            print('%s %f' % (metric, metric_scores[metric]), end='')

        print('\n', end='')
        return summary

    def __init__(self, learning_rate=1e-3, batch_size=1,
                 n_iters=100, name='dnn', valid=None):
        self._batch_size = batch_size
        self._n_iters = n_iters
        self._metrics = {'accuracy': tf.metrics.accuracy}
        self._loss = tf.losses.softmax_cross_entropy
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._name = name
        self._valid = valid

    def fit(self, X, y):

        self._n_classes = y.shape[1]
        # make directories to store training process and models
        os.makedirs(self._name)
        os.makedirs(os.path.join(self._name, 'train'))
        os.makedirs(os.path.join(self._name, 'valid'))

        # input placeholders
        placeholder = {'x': tf.placeholder(tf.float32,
                                           shape=(None, X.shape[1], X.shape[2])),
                       'y': tf.placeholder(tf.int32, shape=(None, y.shape[1]))}

        # connect nn
        with tf.variable_scope('nn') as scope:
            y_prob = self._inference(placeholder['x'])
            loss = self._loss(placeholder['y'], y_prob)
            train_op = self._train(loss)
            tf.summary.scalar('loss', loss)

            # make metric tensors
            metric_tensors = {}
            for metric in self._metrics:
                y_max = tf.reduce_max(y_prob, axis=-1)
                y_pred = tf.cast(tf.equal(y_prob, y_max), dtype=tf.int32)
                metric_tensors[metric], _ = \
                    self._metrics[metric](placeholder['y'], y_pred)

        with tf.Session() as session:
            # prepare summary writer
            summary_writer = \
                {'train': tf.summary.FileWriter(
                    os.path.join(self._name, 'train'), session.graph),
                 'valid': tf.summary.FileWriter(
                     os.path.join(self._name, 'valid'), session.graph)}

            # initialization
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            # Start the training loop.
            for i in range(self._n_iters):
                # train and evaluate train score
                summary = self._iter(session, X, y, train_op,
                                     placeholder, metric_tensors)
                summary_writer['train'].add_summary(summary, i)
                summary_writer['train'].flush()

                # evaluate valid score
                if self._valid is not None:
                    summary = self._iter(session, self._valid['x'],
                                         self._valid['y'], None,
                                         placeholder, metric_tensors)
                    summary_writer['valid'].add_summary(summary, i)
                    summary_writer['valid'].flush()

    def predict(self, X):
        with tf.variable_scope('nn', reuse=True):
            pass
            # X_placeholder = tf.placeholder(
            #     tf.float32, shape=(None, X.shape[1]))
            # logits = self._inference(X_placeholder)
            # y_ = self.sess.run(tf.argmax(logits, axis=1),
            #                    feed_dict={X_placeholder: X})

        # return y_
