import tensorflow as tf
from tqdm import tqdm
from utils import BatchGenerator, make_dir
import os
import pdb


class NNClassifier:
    def _inference(self, X, is_train):
        hidden_size = 30
        word_embedding = tf.constant(self._embedding)

        # embedding.shape = (N_example , N_words, embed_size)
        embedding_b = tf.nn.embedding_lookup(word_embedding, X[:, 0, :])
        embedding_q = tf.nn.embedding_lookup(word_embedding, X[:, 1, :])

        # Encode Body into a LSTM output cell
        with tf.variable_scope('Body_GRU'):
            # forward and backward GRU
            with tf.variable_scope('Body_GRU_fw'):
                cell_b_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
            with tf.variable_scope('Body_GRU_bw'):
                cell_b_bw = tf.nn.rnn_cell.GRUCell(hidden_size)

            # calculate the real length of body
            lengths_b = tf.reduce_sum(tf.sign(X[:, 0, :]), axis=-1)

            # Bi-RNN
            output_b, out_state_b = \
                tf.nn.bidirectional_dynamic_rnn(cell_b_fw,
                                                cell_b_fw,
                                                embedding_b,
                                                sequence_length=lengths_b,
                                                dtype=tf.float32)

            # concat output of RNNs of two directions
            output_b = tf.concat([output_b[0],
                                  output_b[1]], axis=-1,
                                 name='output_b')

            # concate last state of RNNs of two directions
            out_state_b = tf.concat([out_state_b[0], out_state_b[1]], axis=-1)

        with tf.variable_scope('Question_GRU'):
            # forward and backward RNN
            with tf.variable_scope('Question_GRU_fw'):
                cell_q_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
            with tf.variable_scope('Question_GRU_bw'):
                cell_q_bw = tf.nn.rnn_cell.GRUCell(hidden_size)

            # calculate real length of question
            lengths_q = tf.reduce_sum(tf.sign(X[:, 1, :]), axis=-1)

            # Bi-RNN
            output_q, out_state_q = \
                tf.nn.bidirectional_dynamic_rnn(cell_q_fw,
                                                cell_q_fw,
                                                embedding_q,
                                                sequence_length=lengths_q,
                                                dtype=tf.float32)

            # concat output of RNNs of two directions
            output_q = tf.concat([output_q[0], output_q[1]], axis=-1)

            # concate last state of RNNs of two directions
            out_state_q = tf.concat([out_state_q[0], out_state_q[1]],
                                    axis=-1, name='out_state_q')

        similarity = tf.matmul(output_b,
                               tf.reshape(out_state_q, [tf.shape(X)[0], -1, 1]))
        # similarity = tf.reshape(similarity, tf.shape(X))
        mask = tf.reshape(tf.cast(tf.sign(X[:, 0, :]), dtype=tf.float32),
                          [tf.shape(X)[0], -1, 1], name='mask')
        attention_weight = \
            tf.div(tf.exp(similarity) * mask,
                   tf.reshape(tf.reduce_sum(tf.exp(similarity) * mask, axis=1),
                              [tf.shape(X)[0], -1, 1]),
                   name='attention_weight')
        attention_b = tf.reduce_sum(attention_weight * output_b, axis=1)

        output = tf.concat([out_state_q, attention_b], axis=-1)

        output = tf.layers.dense(inputs=output,
                                 units=hidden_size)
        output = tf.nn.elu(output)
        output = tf.layers.dense(inputs=output,
                                 units=hidden_size)
        output = tf.nn.elu(output)
        output = tf.layers.dense(inputs=output,
                                 units=hidden_size)
        output = tf.nn.elu(output)

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
        for b in tqdm(range(X.shape[0] // self._batch_size + 1)):
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
        if train_op == None:
            self._history.append(metrics["accuracy"][0])
            if self._early_stop != None:
                self._history[-1] = max(self._history[-1:-
                                                      self._early_stop - 1: -1])

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
                 n_iters=10, name='dnn', valid=None, embedding=None, early_stop=None):
        self._batch_size = batch_size
        self._n_iters = n_iters
        self._metrics = {'accuracy': tf.metrics.accuracy}
        self._global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._loss = tf.losses.softmax_cross_entropy
        self._name = name
        self._valid = valid
        self._embedding = embedding
        self._history = []
        self._early_stop = early_stop

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
            y_prob = self._inference(placeholder['x'], is_train=True)
            loss = self._loss(placeholder['y'], y_prob)
            reg_const = 0.0005
            l2 = reg_const * sum([tf.nn.l2_loss(tf_var)
                                  for tf_var in tf.trainable_variables()])
            loss += l2
            yp = tf.cast(placeholder['y'], tf.float32)

            train_op = self._optimizer.minimize(
                loss, global_step=self._global_step)

            # make metric tensors
            metric_tensors = {}
            for metric in self._metrics:
                y_pred_argmax = tf.argmax(y_prob, axis=-1)
                y_true_argmax = tf.argmax(placeholder['y'], axis=-1)
                metric_tensors[metric] = \
                    self._metrics[metric](y_true_argmax, y_pred_argmax)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        self._session = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options))
        # prepare summary writer
        summary_writer = \
            {'train': tf.summary.FileWriter(
                os.path.join(self._name, 'train'), self._session.graph),
             'valid': tf.summary.FileWriter(
                 os.path.join(self._name, 'valid'), self._session.graph)}

        # initialization
        self._session.run(tf.global_variables_initializer())
        # checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint/'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self._session, ckpt.model_checkpoint_path)
        initial_step = self._global_step.eval(session=self._session)
        # Start the training loop.
        for i in range(initial_step, self._n_iters):
            # train and evaluate train score
            print('training %i' % i)
            summary = self._iter(X, y, loss, train_op,
                                 placeholder, metric_tensors)
            summary_writer['train'].add_summary(summary, i)
            summary_writer['train'].flush()
            saver.save(self._session, 'checkpoint/session', i)
            # evaluate valid score
            if self._valid is not None:
                print('evaluating %i' % i)
                summary = self._iter(self._valid['x'],
                                     self._valid['y'], loss, None,
                                     placeholder, metric_tensors)
                summary_writer['valid'].add_summary(summary, i)
                summary_writer['valid'].flush()
                if self._early_stop is not None:
                    print(len(self._history))
                    if len(self._history) > self._early_stop and self._history[-1] == self._history[-1 - self._early_stop]:
                        print(self._history)
                        return

    def predict(self, X, prob=False, remove_s=False):
        with tf.variable_scope('nn', reuse=True):
            X_placeholder = tf.placeholder(
                tf.int32, shape=(None, X.shape[1], X.shape[2]))
            # y_prob.shape = (batch_size, n_class)
            y_prob = self._inference(X_placeholder, is_train=False)

            if remove_s:
                y_prob = tf.slice(y_prob, [0, 0], [X.shape[0], 5])
            y_prob = tf.nn.softmax(y_prob)
            # y_prob.shape = (batch_size)
            y_max = tf.reduce_max(y_prob, axis=-1)
            y_pred = tf.cast(
                tf.equal(y_prob, tf.reshape(y_max, (-1, 1))), dtype=tf.int32)

            y_, y_prob = self._session.run([y_pred, y_prob],
                                           feed_dict={X_placeholder: X})
        if not prob:
            return y_
        else:
            return y_, y_prob
