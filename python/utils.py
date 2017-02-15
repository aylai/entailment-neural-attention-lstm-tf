from gensim.models import word2vec as Word2Vec
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import math
from network import TensorFlowTrainable
import progressbar
import time
import datetime

# tokenize punctuation (separate from words with whitespace)
def clean_sequence_to_words(sequence):
    sequence = sequence.lower()

    punctuations = [".", ",", ";", "!", "?", "/", '"', "'", "(", ")", "{", "}", "[", "]", "="]
    for punctuation in punctuations:
        sequence = sequence.replace(punctuation, " {} ".format(punctuation))
    sequence = sequence.replace("  ", " ")
    sequence = sequence.replace("   ", " ")
    sequence = sequence.split(" ")

    todelete = ["", " ", "  "]
    for i, elt in enumerate(sequence):
        if elt in todelete:
            sequence.pop(i)
    return sequence


def load_data(data_dir="/disk2/datasets/snli/snli_1.0/", word2vec_path="/disk2/datasets/word2vec/GoogleNews-vectors-negative300.bin"):

    print "\nLoading word2vec:"
    word2vec = {}
    #word2vec = Word2Vec.Word2Vec.load_word2vec_format(word2vec_path, binary=True)
    if not word2vec:
        print "*******\n******\n*******\nWORD2VEC NOT LOADED\n*******\n*******\n*******"
    print "word2vec: done"

    dataset = {}
    print "\nLoading dataset:"
    for type_set in ["train", "dev", "test"]: 
        df = pd.read_csv(os.path.join(data_dir, "snli_1.0_{}.txt".format(type_set)), delimiter="\t")
        dataset[type_set] = {"premises": df[["sentence1"]].values, "hypothesis": df[["sentence2"]].values, "targets": df[["gold_label"]].values}

    tokenized_dataset = simple_preprocess(dataset=dataset, word2vec=word2vec)
    print "dataset: done\n"
    return word2vec, tokenized_dataset


def simple_preprocess(dataset, word2vec):
    tokenized_dataset = dict((type_set, {"premises": [], "hypothesis": [], "targets": [], "premise_lengths": [], "hyp_lengths": [], "labels": []}) for type_set in dataset)
    print "tokenization:"
    for type_set in dataset:
        print "type_set:", type_set
        map_targets = {"neutral": 0, "entailment": 1, "contradiction": 2}
        target_list = ['neutral','entailment','contradiction']
        num_ids = len(dataset[type_set]["targets"])
        print "num_ids", num_ids
        for i in range(num_ids):
            try:
                premises_tokens = [word for word in clean_sequence_to_words(dataset[type_set]["premises"][i][0])]
                hypothesis_tokens = [word for word in clean_sequence_to_words(dataset[type_set]["hypothesis"][i][0])]
                target = map_targets[dataset[type_set]["targets"][i][0]]
            except:
                pass
            else:
                tokenized_dataset[type_set]["premises"].append(premises_tokens)
                tokenized_dataset[type_set]["premise_lengths"].append(len(premises_tokens))
                tokenized_dataset[type_set]["hypothesis"].append(hypothesis_tokens)
                tokenized_dataset[type_set]["hyp_lengths"].append(len(hypothesis_tokens) + 1)
                tokenized_dataset[type_set]["targets"].append(target)
                L_temp = np.zeros(shape=[len(target_list)], dtype=np.float32)
                L_temp[target] = 1
                tokenized_dataset[type_set]["labels"].append(L_temp)
            sys.stdout.write("\rid: {}/{}      ".format(i + 1, num_ids))
            sys.stdout.flush()
        print ""
    print "tokenization: done"
    return tokenized_dataset


from network import RNN, LSTMCell, AttentionLSTMCell
from batcher import Batcher


def train_old(word2vec, dataset, parameters):
    modeldir = os.path.join(parameters["runs_dir"], parameters["model_name"])
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    logdir = os.path.join(modeldir, "log")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logdir_train = os.path.join(logdir, "train")
    if not os.path.exists(logdir_train):
        os.mkdir(logdir_train)
    logdir_test = os.path.join(logdir, "test")
    if not os.path.exists(logdir_test):
        os.mkdir(logdir_test)
    logdir_dev = os.path.join(logdir, "dev")
    if not os.path.exists(logdir_dev):
        os.mkdir(logdir_dev)
    savepath = os.path.join(modeldir, "save")

    device_string = "/gpu:{}".format(parameters["gpu"]) if parameters["gpu"] else "/cpu:0"
    with tf.device(device_string):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config_proto = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        sess = tf.Session(config=config_proto)

        premises_ph = tf.placeholder(tf.float32, shape=[parameters["sequence_length"], None, parameters["embedding_dim"]], name="premises")
        hypothesis_ph = tf.placeholder(tf.float32, shape=[parameters["sequence_length"], None, parameters["embedding_dim"]], name="hypothesis")
        targets_ph = tf.placeholder(tf.int32, shape=[None], name="targets")
        keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob")

        _projecter = TensorFlowTrainable()
        projecter = _projecter.get_4Dweights(filter_height=1, filter_width=parameters["embedding_dim"], in_channels=1, out_channels=parameters["num_units"], name="projecter")

        #optimizer = tf.train.AdamOptimizer(learning_rate=parameters["learning_rate"], name="ADAM", beta1=0.9, beta2=0.999)
        with tf.variable_scope(name_or_scope="premise"):
            premise = RNN(cell=LSTMCell, num_units=parameters["num_units"], embedding_dim=parameters["embedding_dim"], projecter=projecter, keep_prob=keep_prob_ph)
            premise.process(sequence=premises_ph)

        with tf.variable_scope(name_or_scope="hypothesis"):
            hypothesis = RNN(cell=AttentionLSTMCell, num_units=parameters["num_units"], embedding_dim=parameters["embedding_dim"], hiddens=premise.hiddens, states=premise.states, projecter=projecter, keep_prob=keep_prob_ph)
            hypothesis.process(sequence=hypothesis_ph)

        loss, loss_summary, accuracy, accuracy_summary = hypothesis.loss(targets=targets_ph)

        weight_decay = tf.reduce_sum([tf.reduce_sum(parameter) for parameter in premise.parameters + hypothesis.parameters])

        global_loss = loss + parameters["weight_decay"] * weight_decay

        train_summary_op = tf.merge_summary([loss_summary, accuracy_summary])
        #train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_writer = tf.train.SummaryWriter(logdir_train, sess.graph)
        #train_summary_writer = tf.summary.FileWriter(logdir_train, sess.graph)
        #test_summary_op = tf.merge_summary([loss_summary, accuracy_summary])
        dev_summary_op = tf.merge_summary([loss_summary, accuracy_summary])
        #test_summary_writer = tf.train.SummaryWriter(logdir_test)
        dev_summary_writer = tf.train.SummaryWriter(logdir_dev)

        
        saver = tf.train.Saver(max_to_keep=10)
        #summary_writer = tf.train.SummaryWriter(logdir)
        tf.train.write_graph(sess.graph_def, modeldir, "graph.pb", as_text=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=parameters["learning_rate"], name="ADAM", beta1=0.9, beta2=0.999)
        train_op = optimizer.minimize(global_loss)

        sess.run(tf.initialize_all_variables())
        #sess.run(tf.global_variables_initializer())

        batcher = Batcher(word2vec=word2vec, settings=parameters)
        train_split = "train"
        train_batches = batcher.batch_generator(dataset=dataset[train_split], num_epochs=parameters["num_epochs"], batch_size=parameters["batch_size"]["train"], sequence_length=parameters["sequence_length"])
        print("train data size: %d" % len(dataset["train"]["targets"]))
        num_step_by_epoch = int(math.ceil(len(dataset[train_split]["targets"]) / parameters["batch_size"]["train"]))
        best_dev_accuracy = 0
        for train_step, (train_batch, epoch) in enumerate(train_batches):
            feed_dict = {
                            premises_ph: np.transpose(train_batch["premises"], (1, 0, 2)),
                            hypothesis_ph: np.transpose(train_batch["hypothesis"], (1, 0, 2)),
                            targets_ph: train_batch["targets"],
                            keep_prob_ph: parameters["keep_prob"],
                        }

            _, summary_str, train_loss, train_accuracy = sess.run([train_op, train_summary_op, loss, accuracy], feed_dict=feed_dict)
            train_summary_writer.add_summary(summary_str, train_step)
            if train_step % 100 == 0:
                sys.stdout.write("\rTRAIN | epoch={0}/{1}, step={2}/{3} | loss={4:.2f}, accuracy={5:.2f}%   ".format(epoch + 1, parameters["num_epochs"], train_step % num_step_by_epoch, num_step_by_epoch, train_loss, 100. * train_accuracy))
                sys.stdout.flush()
            if train_step % 5000 == 0:
                dev_batches = batcher.batch_generator(dataset=dataset["dev"], num_epochs=1, batch_size=parameters["batch_size"]["dev"], sequence_length=parameters["sequence_length"])
                for dev_step, (dev_batch, _) in enumerate(dev_batches):
                    feed_dict = {
                                    premises_ph: np.transpose(dev_batch["premises"], (1, 0, 2)),
                                    hypothesis_ph: np.transpose(dev_batch["hypothesis"], (1, 0, 2)),
                                    targets_ph: dev_batch["targets"],
                                    keep_prob_ph: 1.,
                                }

                    summary_str, dev_loss, dev_accuracy = sess.run([dev_summary_op, loss, accuracy], feed_dict=feed_dict)
                    print"\nDEV | loss={0:.2f}, accuracy={1:.2f}%   ".format(dev_loss, 100. * dev_accuracy)
                    print ""
                    dev_summary_writer.add_summary(summary_str, train_step)
                    if dev_accuracy > best_dev_accuracy:
                        saver.save(sess, save_path=savepath+'_best', global_step=train_step)
                    break
            if train_step % 5000 == 0:
                saver.save(sess, save_path=savepath, global_step=train_step)
        print ""


def train(word2vec, dataset, parameters):
    modeldir = os.path.join(parameters["runs_dir"], parameters["model_name"])
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    logdir = os.path.join(modeldir, "log")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logdir_train = os.path.join(logdir, "train")
    if not os.path.exists(logdir_train):
        os.mkdir(logdir_train)
    logdir_test = os.path.join(logdir, "test")
    if not os.path.exists(logdir_test):
        os.mkdir(logdir_test)
    logdir_dev = os.path.join(logdir, "dev")
    if not os.path.exists(logdir_dev):
        os.mkdir(logdir_dev)
    savepath = os.path.join(modeldir, "save")

    device_string = "/gpu:{}".format(parameters["gpu"]) if parameters["gpu"] else "/cpu:0"
    with tf.device(device_string):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config_proto = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        sess = tf.Session(config=config_proto)

        premises_ph = tf.placeholder(tf.float32,
                                     shape=[parameters["sequence_length"], None, parameters["embedding_dim"]],
                                     name="premises")
        hypothesis_ph = tf.placeholder(tf.float32,
                                       shape=[parameters["sequence_length"], None, parameters["embedding_dim"]],
                                       name="hypothesis")
        targets_ph = tf.placeholder(tf.int32, shape=[None], name="targets")
        keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob")

        _projecter = TensorFlowTrainable()
        projecter = _projecter.get_4Dweights(filter_height=1, filter_width=parameters["embedding_dim"], in_channels=1,
                                             out_channels=parameters["num_units"], name="projecter")

        # optimizer = tf.train.AdamOptimizer(learning_rate=parameters["learning_rate"], name="ADAM", beta1=0.9, beta2=0.999)
        with tf.variable_scope(name_or_scope="premise"):
            premise = RNN(cell=LSTMCell, num_units=parameters["num_units"], embedding_dim=parameters["embedding_dim"],
                          projecter=projecter, keep_prob=keep_prob_ph)
            premise.process(sequence=premises_ph)

        with tf.variable_scope(name_or_scope="hypothesis"):
            hypothesis = RNN(cell=AttentionLSTMCell, num_units=parameters["num_units"],
                             embedding_dim=parameters["embedding_dim"], hiddens=premise.hiddens, states=premise.states,
                             projecter=projecter, keep_prob=keep_prob_ph)
            hypothesis.process(sequence=hypothesis_ph)

        loss, loss_summary, accuracy, accuracy_summary = hypothesis.loss(targets=targets_ph)

        weight_decay = tf.reduce_sum(
            [tf.reduce_sum(parameter) for parameter in premise.parameters + hypothesis.parameters])

        global_loss = loss + parameters["weight_decay"] * weight_decay

        train_summary_op = tf.merge_summary([loss_summary, accuracy_summary])
        # train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_writer = tf.train.SummaryWriter(logdir_train, sess.graph)
        # train_summary_writer = tf.summary.FileWriter(logdir_train, sess.graph)
        # test_summary_op = tf.merge_summary([loss_summary, accuracy_summary])
        dev_summary_op = tf.merge_summary([loss_summary, accuracy_summary])
        # test_summary_writer = tf.train.SummaryWriter(logdir_test)
        dev_summary_writer = tf.train.SummaryWriter(logdir_dev)

        saver = tf.train.Saver(max_to_keep=10)
        # summary_writer = tf.train.SummaryWriter(logdir)
        tf.train.write_graph(sess.graph_def, modeldir, "graph.pb", as_text=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=parameters["learning_rate"], name="ADAM", beta1=0.9,
                                           beta2=0.999)
        train_op = optimizer.minimize(global_loss)

        sess.run(tf.initialize_all_variables())
        # sess.run(tf.global_variables_initializer())

        batcher = Batcher(word2vec=word2vec, settings=parameters)
        #train_split = "train"
        #train_batches = batcher.batch_generator(dataset=dataset[train_split], num_epochs=parameters["num_epochs"],
                                               # batch_size=parameters["batch_size"]["train"],
                                               # sequence_length=parameters["sequence_length"])
        #print("train data size: %d" % len(dataset["train"]["targets"]))
        #num_step_by_epoch = int(math.ceil(len(dataset[train_split]["targets"]) / parameters["batch_size"]["train"]))
        #best_dev_accuracy = 0
        print("train data size: %d" % len(dataset["train"]["targets"]))
        best_dev_accuracy = 0.0
        total_loss = 0.0
        timestamp = time.time()
        for epoch in range(parameters["num_epochs"]):
            print("epoch %d" % epoch)
            train_batches = batcher.batch_generator(dataset=dataset["train"],
                                                         num_epochs=1,
                                                         batch_size=parameters["batch_size"]["train"],
                                                         sequence_length=parameters["sequence_length"])
            steps = len(dataset["train"]["targets"]) / parameters["batch_size"]["train"]

            # progress bar http://stackoverflow.com/a/3002114
            bar = progressbar.ProgressBar(maxval=steps / 10 + 1,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for step, (train_batch, train_epoch) in enumerate(train_batches):
                feed_dict = {
                    premises_ph: np.transpose(train_batch["premises"], (1, 0, 2)),
                    hypothesis_ph: np.transpose(train_batch["hypothesis"], (1, 0, 2)),
                    targets_ph: train_batch["targets"],
                    keep_prob_ph: parameters["keep_prob"],
                }
                _, summary_str, train_loss, train_accuracy = sess.run([train_op, train_summary_op, loss, accuracy],
                                                                      feed_dict=feed_dict)
                total_loss += train_loss
                train_summary_writer.add_summary(summary_str, step)
                if step % 100 == 0:  # eval 1 random dev batch
                    # eval 1 random dev batch
                    dev_batches = batcher.batch_generator(dataset=dataset["dev"], num_epochs=1,
                                                          batch_size=parameters["batch_size"]["dev"],
                                                          sequence_length=parameters["sequence_length"])
                    for dev_step, (dev_batch, _) in enumerate(dev_batches):
                        feed_dict = {
                            premises_ph: np.transpose(dev_batch["premises"], (1, 0, 2)),
                            hypothesis_ph: np.transpose(dev_batch["hypothesis"], (1, 0, 2)),
                            targets_ph: dev_batch["targets"],
                            keep_prob_ph: 1.,
                        }

                        summary_str, dev_loss, dev_accuracy = sess.run([dev_summary_op, loss, accuracy],
                                                                       feed_dict=feed_dict)
                        dev_summary_writer.add_summary(summary_str, step)
                        break
                    bar.update(step / 10 + 1)
            bar.finish()
            # eval on all dev
            dev_batches = batcher.batch_generator(dataset=dataset["dev"], num_epochs=1,
                                                  batch_size=len(dataset["dev"]["targets"]),
                                                  sequence_length=parameters["sequence_length"])
            dev_accuracy = 0
            for dev_step, (dev_batch, _) in enumerate(dev_batches):
                feed_dict = {
                    premises_ph: np.transpose(dev_batch["premises"], (1, 0, 2)),
                    hypothesis_ph: np.transpose(dev_batch["hypothesis"], (1, 0, 2)),
                    targets_ph: dev_batch["targets"],
                    keep_prob_ph: 1.,
                }
                summary_str, dev_loss, dev_accuracy = sess.run([dev_summary_op, loss, accuracy],
                                                               feed_dict=feed_dict)
                print"\nDEV full | loss={0:.2f}, accuracy={1:.2f}%   ".format(dev_loss, 100. * dev_accuracy)
                print ""
                if dev_accuracy > best_dev_accuracy:
                    saver.save(sess, save_path=savepath + '_best', global_step=(epoch+1)*steps)
                break
            saver.save(sess, save_path=savepath, global_step=(epoch+1)*steps)
            current_time = time.time()
            print("Iter %3d  Loss %-8.3f  Dev Acc %-6.2f  Time %-5.2f at %s" %
                  (epoch, total_loss, dev_accuracy,
                   (current_time - timestamp) / 60.0, str(datetime.datetime.now())))
            total_loss = 0.0
        print ""

def test(word2vec, dataset, parameters, loadpath):
    print "1"
    device_string = "/gpu:{}".format(parameters["gpu"]) if parameters["gpu"] else "/cpu:0"
    with tf.device(device_string):
        print "2"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config_proto = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        sess = tf.Session(config=config_proto)

        premises_ph = tf.placeholder(tf.float32,
                                     shape=[parameters["sequence_length"], None, parameters["embedding_dim"]],
                                     name="premises")
        hypothesis_ph = tf.placeholder(tf.float32,
                                       shape=[parameters["sequence_length"], None, parameters["embedding_dim"]],
                                       name="hypothesis")
        targets_ph = tf.placeholder(tf.int32, shape=[None], name="targets")
        keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob")

        _projecter = TensorFlowTrainable()
        projecter = _projecter.get_4Dweights(filter_height=1, filter_width=parameters["embedding_dim"], in_channels=1,
                                             out_channels=parameters["num_units"], name="projecter")

        with tf.variable_scope(name_or_scope="premise"):
            premise = RNN(cell=LSTMCell, num_units=parameters["num_units"], embedding_dim=parameters["embedding_dim"],
                          projecter=projecter, keep_prob=keep_prob_ph)
            premise.process(sequence=premises_ph)

        with tf.variable_scope(name_or_scope="hypothesis"):
            hypothesis = RNN(cell=AttentionLSTMCell, num_units=parameters["num_units"],
                             embedding_dim=parameters["embedding_dim"], hiddens=premise.hiddens, states=premise.states,
                             projecter=projecter, keep_prob=keep_prob_ph)
            hypothesis.process(sequence=hypothesis_ph)

        loss, loss_summary, accuracy, accuracy_summary = hypothesis.loss(targets=targets_ph)

        loader = tf.train.Saver()
        loader.restore(sess, loadpath)

        batcher = Batcher(word2vec=word2vec, settings=parameters)
        test_batches = batcher.batch_generator(dataset=dataset["test"], num_epochs=1,
                                               batch_size=parameters["batch_size"]["test"],
                                               sequence_length=parameters["sequence_length"])
        print "2.5"
        for test_step, (test_batch, _) in enumerate(test_batches):
            print "3"
            feed_dict = {
                premises_ph: np.transpose(test_batch["premises"], (1, 0, 2)),
                hypothesis_ph: np.transpose(test_batch["hypothesis"], (1, 0, 2)),
                targets_ph: test_batch["targets"],
                keep_prob_ph: 1.,
            }

            test_loss, test_accuracy = sess.run([loss, accuracy],
                                                             feed_dict=feed_dict)
            print"\nTEST | loss={0:.2f}, accuracy={1:.2f}%   ".format(test_loss, 100. * test_accuracy)
            print ""