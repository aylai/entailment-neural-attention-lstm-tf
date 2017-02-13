import tensorflow as tf
import Layer
import numpy as np
import os
import time
import progressbar
import datetime


class Model(object):

    def __init__(self, batcher, dataset, parameters, load_model=None):
        self.batcher = batcher
        self.dataset = dataset
        self.parameters = parameters

        """ Model Definition """
        self.premise_lengths = tf.placeholder(tf.int32, shape=[None], name="premise_lengths")
        self.hyp_lengths = tf.placeholder(tf.int32, shape=[None], name="hypothesis_lengths")
        # self.premises_ph = tf.placeholder(tf.float32,
        #                              shape=[parameters["sequence_length"], None, parameters["embedding_dim"]],
        #                              name="premises")
        # self.hypotheses_ph = tf.placeholder(tf.float32,
        #                                shape=[parameters["sequence_length"], None, parameters["embedding_dim"]],
        #                                name="hypothesis")
        self.premises_ph = tf.placeholder(tf.float32,
                                          shape=[None, parameters["sequence_length"], parameters["embedding_dim"]],
                                          name="premises")
        self.hypotheses_ph = tf.placeholder(tf.float32,
                                            shape=[None, parameters["sequence_length"], parameters["embedding_dim"]],
                                            name="hypothesis")
        self.labels_ph = tf.placeholder(tf.int32, shape=[None, parameters["num_classes"]], name="labels")

        sentence_pair = self.sentence_pair_rep()

        # loss
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(sentence_pair, self.labels_ph))
        self.predictions = tf.argmax(sentence_pair, 1)
        # correct_pred = tf.equal(tf.argmax(sentence_pair, 1), tf.argmax(self.labels_ph, 1))
        # self.accuracy = 100.0 * sum(correct_pred) / len(correct_pred)

        optimizer = tf.train.AdamOptimizer(self.parameters["learning_rate"])
        self.train_op = optimizer.minimize(self.loss)

        modeldir = os.path.join(parameters["runs_dir"], parameters["model_name"])
        if not os.path.exists(modeldir):
            os.mkdir(modeldir)
        self.logdir = os.path.join(modeldir, "log")
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)
        self.savepath = os.path.join(modeldir, "save")

        # Tensorboard
        self.train_cost_summary = tf.scalar_summary('train_cost', self.loss)
        self.dev_cost_summary = tf.scalar_summary('dev_cost', self.loss)
        self.writer = tf.train.SummaryWriter(self.logdir, graph=tf.get_default_graph())
        self.saver = tf.train.Saver()

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        if load_model is not None:
            print "Restoring ", load_model
            self.saver.restore(self.sess, load_model)
        else:
            print "Initializing"
            self.sess.run(tf.initialize_all_variables())

    def sentence_pair_rep(self):

        lstm = tf.nn.rnn_cell.LSTMCell(self.parameters["num_units"], state_is_tuple=True)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.parameters["keep_prob"])

        output_layer1 = Layer.ff_w(2 * self.parameters["num_units"], self.parameters["output_dim"], name='Output1')
        output_bias1 = Layer.ff_b(self.parameters["output_dim"], 'OutputBias1')
        output_layer2 = Layer.ff_w(self.parameters["output_dim"], self.parameters["output_dim"], 'Output2')
        output_bias2 = Layer.ff_b(self.parameters["output_dim"], 'OutputBias2')
        output_layer3 = Layer.ff_w(self.parameters["output_dim"], self.parameters["num_classes"], 'Output3')
        output_bias3 = Layer.ff_b(self.parameters["num_classes"], 'OutputBias3')

        outputs1, fstate1 = tf.nn.dynamic_rnn(lstm, self.premises_ph, sequence_length=self.premise_lengths,
                                              dtype=tf.float32)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            outputs2, fstate2 = tf.nn.dynamic_rnn(lstm, self.hypotheses_ph, sequence_length=self.hyp_lengths,
                                              dtype=tf.float32)

        logits1 = tf.nn.dropout(
            tf.nn.tanh(tf.matmul(tf.concat(1, [fstate1[0], fstate2[0]]), output_layer1)) + output_bias1,
            self.parameters["keep_prob"])

        logits2 = tf.nn.dropout(
            tf.nn.tanh(tf.matmul(logits1, output_layer2) + output_bias2), self.parameters["keep_prob"])

        logits3 = tf.nn.tanh(tf.matmul(logits2, output_layer3) + output_bias3)
        return logits3

    def train(self):
        print("train data size: %d" % len(self.dataset["train"]["targets"]))
        best_dev_accuracy = 0.0
        total_loss = 0.0
        timestamp = time.time()
        for epoch in range(self.parameters["num_epochs"]):
            print("new epoch %d" %epoch)
            train_batches = self.batcher.batch_generator(dataset=self.dataset["train"],
                                                         num_epochs=1,
                                                         batch_size=self.parameters["batch_size"]["train"],
                                                         sequence_length=self.parameters["sequence_length"])
            steps = len(self.dataset["train"]["targets"]) / self.parameters["batch_size"]["train"]

            # progress bar http://stackoverflow.com/a/3002114
            bar = progressbar.ProgressBar(maxval=steps / 10 + 1,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for step, (train_batch, epoch) in enumerate(train_batches):
                feed_dict = {
                    #self.premises_ph: np.transpose(train_batch["premises"], (1, 0, 2)),
                    #self.hypotheses_ph: np.transpose(train_batch["hypothesis"], (1, 0, 2)),
                    self.premises_ph: train_batch["premises"],
                    self.hypotheses_ph: train_batch["hypothesis"],
                    self.labels_ph: train_batch["labels"],
                    self.premise_lengths: train_batch["premise_lengths"],
                    self.hyp_lengths: train_batch["hyp_lengths"],
                }
                _, summary_str, train_loss, train_pred = self.sess.run([self.train_op, self.train_cost_summary, self.loss, self.predictions],
                                                                  feed_dict=feed_dict)
                total_loss += train_loss
                self.writer.add_summary(summary_str, step)
                if step % 100 == 0:  # eval 1 random dev batch
                    self._eval(self.dataset["dev"], "dev", True) # TODO change to false: one dev batch only
                    bar.update(step / 10 + 1)
            bar.finish()
            dev_loss, dev_accuracy = self._eval(self.dataset["dev"], "dev", True)  # eval on full dev for printing (but not tensorboard)
            if dev_accuracy > best_dev_accuracy:
                self.saver.save(self.sess, save_path=self.savepath + '_best', global_step=step)
            self.saver.save(self.sess, save_path=self.savepath, global_step=step)
            current_time = time.time()
            print("Iter %3d  Loss %-8.3f  Dev Acc %-6.2f  Time %-5.2f at %s" %
                  (epoch, total_loss, dev_accuracy,
                   (current_time - timestamp) / 60.0, str(datetime.datetime.now())))
            total_loss = 0.0

    def train_new(self):
        print("train data size: %d" % len(self.dataset["train"]["targets"]))
        best_dev_accuracy = 0.0
        total_loss = 0.0
        timestamp = time.time()
        #for epoch in range(self.parameters["num_epochs"]):
         #   print("new epoch %d" %epoch)
        train_batches = self.batcher.batch_generator(dataset=self.dataset["train"],
                                                         num_epochs=self.parameters["num_epochs"],
                                                         batch_size=self.parameters["batch_size"]["train"],
                                                         sequence_length=self.parameters["sequence_length"])
            #steps = len(self.dataset["train"]["targets"]) / self.parameters["batch_size"]["train"]

            # progress bar http://stackoverflow.com/a/3002114
            #bar = progressbar.ProgressBar(maxval=steps / 10 + 1,
            #                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            #bar.start()
        for step, (train_batch, epoch) in enumerate(train_batches):
            feed_dict = {
                #self.premises_ph: np.transpose(train_batch["premises"], (1, 0, 2)),
                #self.hypotheses_ph: np.transpose(train_batch["hypothesis"], (1, 0, 2)),
                self.premises_ph: train_batch["premises"],
                self.hypotheses_ph: train_batch["hypothesis"],
                self.labels_ph: train_batch["labels"],
                self.premise_lengths: train_batch["premise_lengths"],
                self.hyp_lengths: train_batch["hyp_lengths"],
            }
            _, summary_str, train_loss, train_pred = self.sess.run([self.train_op, self.train_cost_summary, self.loss, self.predictions],
                                                              feed_dict=feed_dict)
            total_loss += train_loss
            self.writer.add_summary(summary_str, step)
            if step % 100 == 0:  # eval 1 random dev batch
                self._eval(self.dataset["dev"], "dev", True) # TODO change to false: one dev batch only
                #bar.update(step / 10 + 1)
            #bar.finish()
            if step % 5000 == 0:
                dev_loss, dev_accuracy = self._eval(self.dataset["dev"], "dev", True)  # eval on full dev for printing (but not tensorboard)
                if dev_accuracy > best_dev_accuracy:
                    self.saver.save(self.sess, save_path=self.savepath + '_best', global_step=step)
                self.saver.save(self.sess, save_path=self.savepath, global_step=step)
                current_time = time.time()
                print("Iter %3d  Loss %-8.3f  Dev Acc %-6.2f  Time %-5.2f at %s" %
                      (epoch, total_loss, dev_accuracy,
                       (current_time - timestamp) / 60.0, str(datetime.datetime.now())))
                #total_loss = 0.0

    def _eval(self, eval_data, data_name, full):
        if full:
            eval_batches = self.batcher.batch_generator(dataset=eval_data, num_epochs=1,
                                               batch_size=self.parameters["batch_size"][data_name],
                                               sequence_length=self.parameters["sequence_length"])
        else:
            eval_batches = self.batcher.batch_generator(dataset=eval_data, num_epochs=1,
                                                      batch_size=len(eval_data["targets"]),
                                                      sequence_length=self.parameters["sequence_length"])
        for step, (eval_batch, epoch) in enumerate(eval_batches):
            feed_dict = {
                #self.premises_ph: np.transpose(eval_batch["premises"], (1, 0, 2)),
                #self.hypotheses_ph: np.transpose(eval_batch["hypothesis"], (1, 0, 2)),
                self.premises_ph: eval_batch["premises"],
                self.hypotheses_ph: eval_batch["hypothesis"],
                self.labels_ph: eval_batch["labels"],
                self.premise_lengths: eval_batch["premise_lengths"],
                self.hyp_lengths: eval_batch["hyp_lengths"],
            }

            summary_str, eval_loss, eval_pred = self.sess.run([self.dev_cost_summary, self.loss, self.predictions],
                                                    feed_dict=feed_dict)
            if not full:
                self.writer.add_summary(summary_str, step)
            return eval_loss, self._accuracy(eval_batch["targets"], eval_pred)

    def _accuracy(self, labels, predictions):
        correct_pred = np.equal(predictions, labels)
        return 100.0 * sum(correct_pred) / len(correct_pred)

    def test(self):
        test_loss, test_accuracy = self._eval(self.dataset["test"], "test", None, True)
        print("Loss %-8.3f  Test Acc %-6.2f" % (test_loss, test_accuracy))
