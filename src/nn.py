import tensorflow as tf
from tqdm import tqdm
from utils import BatchGenerator, make_dir
import os
import pdb


class NNClassifier:
    def _inference(self, X, is_train):
        hidden_size = 50
        dropout = 0.3
        word_embedding = tf.get_variable("embedding", initializer=tf.constant(self._embedding) )
        # X.shape = (batch_size, N_words)
        # embedding.shape = (N_example , N_words, embed_size)
        embedding_b = tf.nn.embedding_lookup(word_embedding, X[:, 0, :])
        embedding_q = tf.nn.embedding_lookup(word_embedding, X[:, 1, :])
        # Encode Body into a LSTM output cell
        
        with tf.variable_scope('Body_GRU'):
            with tf.variable_scope('Body_GRU_fw'):
                cell_b_fw = tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True)
            with tf.variable_scope('Body_GRU_bw'):
                cell_b_bw = tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True)
            # this line to calculate the real length of seq
            # all seq are padded to be of the same length which is N_words
            lengths_b = tf.reduce_sum(tf.sign(X[:, 0, :]), axis=-1)
            output_b, out_state_b = tf.nn.bidirectional_dynamic_rnn(cell_b_fw,
            cell_b_bw, embedding_b, sequence_length=lengths_b, dtype=tf.float32)
            #output_b = tf.concat([output_b[0], output_b[1]], axis=-1)
            #out_state_b = tf.concat([out_state_b[0], out_state_b[1]], axis=-1)
            #output_b = output_b[0]
            #out_state_b = out_state_b[0]

        with tf.variable_scope('Question_GRU'):
            with tf.variable_scope('Question_GRU_fw'):
                cell_q_fw = tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True)
            with tf.variable_scope('Question_GRU_bw'):
                cell_q_bw = tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True)
            lengths_q = tf.reduce_sum(tf.sign(X[:, 1, :]), axis=-1)
            output_q, out_state_q = tf.nn.bidirectional_dynamic_rnn(cell_q_fw,
            cell_q_bw, embedding_q, initial_state_fw=out_state_b[0], 
            initial_state_bw=out_state_b[1], sequence_length=lengths_q, dtype = tf.float32)
        
        # output is [batch_size, N_words(timestep), hidden_size], 
        # we need last timestep output : (batch_size, hidden_size)
        out_state_b = tf.concat([out_state_b[0][1], out_state_b[1][1]], axis = -1)
        out_state_q = tf.concat([out_state_q[0][1], out_state_q[1][1]], axis = -1)
        # output is [batch_size, 2*hidden_size]
        #output = tf.concat([output_b, output_q], axis = -1)
        # (batch_size, _n_classes)
        output = tf.concat([out_state_b, out_state_q], axis = -1)
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
                self._history[-1] = max(self._history[-1:-self._early_stop-1: -1])

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
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self._loss = tf.losses.softmax_cross_entropy
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
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
            yp = tf.cast(placeholder['y'], tf.float32)
            
            #yt = tf.contrib.layers.softmax(y_prob)
            yt = y_prob
            pos = tf.reduce_sum(yp*yt, axis=1)
            neg = tf.reduce_max((1-yp)*yt, reduction_indices=[1])
            #loss = 2-(pos-neg)
            #loss = tf.reduce_sum(tf.maximum(tf.zeros_like(loss), loss) )
            #loss += self._loss(yp, y_prob)
            #loss = tf.maximum(tf.zeros([X.shape[0]]), 0.25-tf.(tf.reduce_sum(yp*y_prob, axis=1)-tf.reduce_max((1-yp)*y_prob, reduction_indices=[1]) ))
            #loss = self._loss(yp, y_prob)
            train_op = self._optimizer.minimize(loss, global_step=self._global_step)

            # make metric tensors
            metric_tensors = {}
            for metric in self._metrics:
                y_pred_argmax = tf.argmax(y_prob, axis=-1)
                y_true_argmax = tf.argmax(placeholder['y'], axis=-1)
                metric_tensors[metric] = \
                    self._metrics[metric](y_true_argmax, y_pred_argmax)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
                    print(len(self._history) )
                    if len(self._history) > self._early_stop and self._history[-1] <= self._history[-1-self._early_stop] :
                        print(self._history)
                        return 

    def predict(self, X, prob=False):
        with tf.variable_scope('nn', reuse=True):
            X_placeholder = tf.placeholder(
                tf.int32, shape=(None, X.shape[1], X.shape[2]))
            # y_prob.shape = (batch_size, n_class)
            y_prob = self._inference(X_placeholder, is_train=False)
            y_prob = tf.nn.softmax(y_prob)
            # y_prob.shape = (batch_size)
            y_max = tf.reduce_max(y_prob, axis=-1)
            y_pred = tf.cast(tf.equal(y_prob, tf.reshape(y_max, (-1, 1))), dtype=tf.int32)

            y_, y_prob = self._session.run([y_pred, y_prob],
                                   feed_dict={X_placeholder: X})
        if not prob:
            return y_
        else:
            return y_, y_prob
