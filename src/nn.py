import tensorflow as tf
from tqdm import tqdm
from utils import BatchGenerator
import os
import pdb


class NNClassifier:
    def _inference(self, X, is_train):
        self.hidden_size = hidden_size = 20
        dropout = 0.3
        word_embedding = tf.constant(self._embedding)
        # X.shape = (batch_size, N_words)
        # embedding.shape = (N_example , N_words, embed_size)
        embedding_b = tf.nn.embedding_lookup(word_embedding, X[:, 0, :])
        embedding_q = tf.nn.embedding_lookup(word_embedding, X[:, 1, :])
        # Encode Body into a LSTM output cell
    
        with tf.variable_scope('Body_GRU'):
            with tf.variable_scope('Body_GRU_fw'):
                cell_b_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
            with tf.variable_scope('Body_GRU_bw'):
                cell_b_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
            # this line to calculate the real length of seq
            # all seq are padded to be of the same length which is N_words
            # output is [batch_size, N_words(timestep), hidden_size]
            lengths_b = tf.reduce_sum(tf.sign(X[:, 0, :]), axis=-1)
            output_b, out_state_b = tf.nn.bidirectional_dynamic_rnn(cell_b_fw,
            cell_b_fw, embedding_b, sequence_length=lengths_b, dtype=tf.float32)
            output_b = tf.concat([output_b[0], output_b[1]], axis=-1)
            out_state_b = tf.concat([out_state_b[0], out_state_b[1]], axis=-1)

        with tf.variable_scope('Question_GRU'):
            with tf.variable_scope('Question_GRU_fw'):
                cell_q_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
            with tf.variable_scope('Question_GRU_bw'):
                cell_q_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
            lengths_q = tf.reduce_sum(tf.sign(X[:, 1, :]), axis=-1)
            output_q, out_state_q = tf.nn.bidirectional_dynamic_rnn(cell_q_fw,
            cell_q_fw, embedding_q, sequence_length=lengths_q, dtype=tf.float32)
            output_q = tf.concat([output_q[0], output_q[1]], axis=-1)
            out_state_q = tf.concat([out_state_q[0], out_state_q[1]], axis=-1)
        

        # output_b is [batch_size, N_words, 2*hidden_size]

        #attention = tf.nn.softmax(tf.reduce_sum(output_b * tf.reshape(out_state_q,[-1,1,hidden_size*2]), axis=-1, keep_dims=True))
        #tmp = attention*output_b
        #attention_b = tf.reduce_sum(tmp, axis=-1)       
        #output = tf.concat([out_state_q,out_state_b, attention_b], axis=-1)
        n_words_q = output_q.shape[1]
        # a tensorarray of state
        # a_array = tf.TensorArray(dtype=tf.float32, size=n_words_q)
        # i = tf.constant(0, dtype=tf.int32)
        # cond = lambda i, b, q, array: tf.less(i, n_words_q)
        # body = lambda i, b, q, array: self._match_lstm(i, b, q, array)
        # res is a (i, output_b, output_q, a_array tuple)
        # res = tf.while_loop(cond=cond, body=body, loop_vars=(i, output_b, output_q, a_array))
        # a_state is a [T, B, hidden_size] -> [B, T, hidden_size] 
        # a_state = tf.transpose(res[-1].stack() ,[1, 0, 2])
        C_d = self._dynamic_coattention(output_b, output_q)
        #  match_lstm rnn, dynamic coattention
        with tf.variable_scope('match_lstm_GRU_dynamic_coattention'):
            with tf.variable_scope('match_gru_fw'):
                cell_m_fw = tf.nn.rnn_cell.GRUCell(2*hidden_size)
            with tf.variable_scope('match_gru_bw'):
                cell_m_bw = tf.nn.rnn_cell.GRUCell(2*hidden_size)
            lengths_q = tf.reduce_sum(tf.sign(X[:, 1, :]), axis=-1)
            output_m, out_state_m = tf.nn.bidirectional_dynamic_rnn(cell_m_fw,
            cell_m_fw, C_d, sequence_length=lengths_q, dtype=tf.float32)
            output_m = tf.concat([output_m[0], output_m[1]], axis=-1)
            out_state_m = tf.concat([out_state_m[0], out_state_m[1]], axis=-1)
        output = tf.concat([out_state_m, out_state_q, out_state_b], axis=-1)
        output = tf.layers.dense(inputs=output,
                                units=10)
        output = tf.nn.relu(output)
    
        dense = tf.layers.dense(inputs=output,
                                units=self._n_classes)
        
        return dense
    def _dynamic_coattention(self, output_b, output_q):
        with tf.variable_scope('dynamic_coattention'):
            # L is shape of [batch_size, timestep(b), timestep(q)]
            timestep = tf.shape(output_q)[1]
            batch_size = tf.shape(output_q)[0]
            L = tf.matmul(output_b, tf.transpose(output_q, [0, 2, 1]))
            # A_q is shape of [batch_size, timestep(b), timestep(q)]
            A_q = tf.nn.softmax(L, dim=2)
            # A_b is shape of [batch_size, timestep(q), timestep(b)]
            A_b = tf.transpose(tf.nn.softmax(L, dim=1), [0, 2, 1])
            # C_q is shape of [batch_size, timestep(q), hidden_size]
            C_q = tf.matmul(tf.transpose(A_q, [0, 2, 1]), output_b)
            # C_b is shape of [batch_size, timestep(b), hidden_size]
            C_b = tf.matmul(tf.transpose(A_b, [0, 2, 1]), output_q)
            # C_d is shape of [batch_size, timestep(b), hidden_size * 2]
            C_d = tf.matmul(tf.transpose(A_b, [0, 2, 1]), tf.concat([output_q, C_q], axis=-1))
        return C_d

    def _match_lstm(self, i, output_b, output_q, a_array):
        # Retrieve output_b_i with shape of [batch_size, 1(ith word), 2*hidden_size]
        hidden_size = self.hidden_size * 2
        timestep = tf.shape(output_b)[1]
        output_q_i = tf.reshape(output_q[:, i, :], [-1, 1, hidden_size])
        #output_q_i = tf.slice(output_q, begin=[0, i, 0], size=[self._batch_size, 1, hidden_size])
        #print(output_q_i.shape)
        #print(output_b)
        with tf.variable_scope("match_lstm"):
            we = tf.get_variable(name="we", shape=[1, 1 ,hidden_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0.0, 1))
            W_s = tf.get_variable(name="W_s", shape=[hidden_size, hidden_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0.0, 1))
            W_t = tf.get_variable(name="W_t", shape=[hidden_size, hidden_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0.0, 1))
            #W_m = tf.get_variable(name="W_m")
            temp1 = tf.reshape(tf.matmul(tf.reshape(output_b, [-1, hidden_size]), W_s), 
                [-1, timestep, hidden_size])
            temp2 = tf.reshape(tf.matmul(tf.reshape(output_q_i,[-1, hidden_size]), W_t),
                [-1, 1, hidden_size])
            middle = tf.tanh(temp1 + temp2)
            # reduce hidden_size dimension -> [B, T, 1]
            e_i = tf.reduce_sum(we * middle, axis=-1, keep_dims=True)
            # softmax [B, T, 1] 'T' dimenstion
            a_i = tf.nn.softmax(e_i, dim=1)
            # weighted sum of output_b([B, T, 1] * [B, T, hidden_size]) -> [B, hidden_size]
            a_i = tf.reduce_sum(a_i * output_b, axis=1, keep_dims=False)
            #print(a_i)
            a_array = a_array.write(i, tf.concat([a_i, tf.reshape(output_q_i, [-1, hidden_size])], axis=-1))
        i = tf.add(i, 1)
        return i, output_b, output_q, a_array
    def _attention(self, inputs, attention_size, time_major=False, return_alphas=False):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
         for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        """

        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        inputs_shape = inputs.shape
        #print(inputs.shape)
        sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

        # Attention mechanism
        with tf.variable_scope('attention'):
            W_omega = tf.get_variable(name="W", shape=[hidden_size, attention_size], dtype=tf.float32,
                initializer=tf.random_normal_initializer(0.0, 0.1))
            b_omega = tf.get_variable(name="b", shape=[attention_size], dtype=tf.float32,
                initializer=tf.random_normal_initializer(0.0, 0.1))
            u_omega = tf.get_variable(name="u", shape=[attention_size], dtype=tf.float32,
                initializer=tf.random_normal_initializer(0.0, 0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas
        
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
        os.mkdir(self._name)
        os.mkdir(os.path.join(self._name, 'train'))
        os.mkdir(os.path.join(self._name, 'valid'))

        # input placeholders
        placeholder = {'x': tf.placeholder(tf.int32,
                                           shape=(None, X.shape[1], X.shape[2])),
                       'y': tf.placeholder(tf.int32, shape=(None, y.shape[1]))}

        # connect nn
        with tf.variable_scope('nn') as scope:
            y_prob = self._inference(placeholder['x'], is_train=True)
            loss = self._loss(placeholder['y'], y_prob)
            reg_const  = 0.0005
            l2 = reg_const * sum([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() ])
            #loss += l2
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
            train_op = self._optimizer.minimize(loss)
            add_global_step_op = tf.assign(self._global_step, self._global_step+1)
            # make metric tensors
            metric_tensors = {}
            for metric in self._metrics:
                y_pred_argmax = tf.argmax(y_prob, axis=-1)
                y_true_argmax = tf.argmax(placeholder['y'], axis=-1)
                metric_tensors[metric] = \
                    self._metrics[metric](y_true_argmax, y_pred_argmax)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
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
        print("initial step: {}".format(initial_step))
        # Start the training loop.
        for i in range(initial_step, self._n_iters):
            # train and evaluate train score
            print('training %i' % i)
            _ = self._session.run([add_global_step_op])
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
                    if len(self._history) > self._early_stop and self._history[-1] == self._history[-1-self._early_stop] :
                        print(self._history)
                        return 

    def predict(self, X, prob=False, remove_s = False):
        with tf.variable_scope('nn', reuse=True):
            X_placeholder = tf.placeholder(
                tf.int32, shape=(None, X.shape[1], X.shape[2]))
            # y_prob.shape = (batch_size, n_class)
            y_prob = self._inference(X_placeholder, is_train=False)

            if remove_s:
                y_prob = tf.slice(y_prob, [0,0], [X.shape[0],5])
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
