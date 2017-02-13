import argparse
import os
from utils import load_data, train, test
import time
import LSTM
from batcher import Batcher
import sys

dirname, filename = os.path.split(os.path.abspath(__file__))
GIT_DIR = "/".join(dirname.split("/")[:-1])

PYTHON_DIR = os.path.join(GIT_DIR, "python")
DATA_DIR = os.path.join(GIT_DIR, "data")
RUNS_DIR = os.path.join(GIT_DIR, "runs")


if __name__ == "__main__":

    # ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="active this flag to train the model")
    parser.add_argument("--test", action="store_true", help="active this flag to test the model")
    parser.add_argument("--test_split", default="test", help="data split to evaluate")
    parser.add_argument("--model_path", help="path to saved model")
    parser.add_argument("--data_dir", default=os.path.join(DATA_DIR, "dataset"), help="path to the SNLI dataset directory")
    parser.add_argument("--word2vec_path", default="/home/aylai2/data/word2vec/GoogleNews-vectors-negative300.bin", help="path to the pretrained Word2Vect .bin file")
    parser.add_argument("--model_name", type=str, default="attention_lstm")
    parser.add_argument("--model_type", type=str, default="attention")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--keep_prob", type=float, default=0.8)
    parser.add_argument("--batch_size_train", type=int, default=30)
    parser.add_argument("--batch_size_dev", type=int, default=30)
    parser.add_argument("--batch_size_test", type=int, default=10000)
    parser.add_argument("--gpu", type=str, default="0", help="set gpu to '' to use CPU mode")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=300, help="Word2Vec dimension")
    parser.add_argument("--sequence_length", type=int, default=30, help="final length of each sequence (premise and hypothesis), padded with null-words if needed")
    parser.add_argument("--num_units", type=int, default=100, help="LSTM output dimension (k in the original paper)")
    parser.add_argument("--output_dim", type=int, default=200, help="output dimension of ff layers for LSTM (Bowman)")
    args = parser.parse_args()

    # PARAMETERS
    parameters = {
                    "runs_dir": RUNS_DIR,
                    "embedding_dim": args.embedding_dim,
                    "num_units": args.num_units,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "keep_prob": args.keep_prob,
                    "model_name": args.model_name,
                    "gpu": args.gpu or None,
                    "batch_size": {"train": args.batch_size_train, "dev": args.batch_size_dev, "test": args.batch_size_test},
                    "sequence_length": args.sequence_length,
                    "weight_decay": args.weight_decay,
                    "num_classes": args.num_classes,
                    "output_dim": args.output_dim,
                    "settings": 'mine',
                }

    for key, parameter in parameters.iteritems():
        print "{}: {}".format(key, parameter)

    if not args.train and not args.test:
        print "Choose to train or test model."
        sys.exit(0)

    start = time.time()

    # MAIN
    word2vec, dataset = load_data(data_dir=args.data_dir, word2vec_path=args.word2vec_path)

    if args.model_type == "attention":
        if args.train:
            train(word2vec=word2vec, dataset=dataset, parameters=parameters)

        if args.test:
            print "here"
            test(word2vec=word2vec, dataset=dataset, parameters=parameters, loadpath=args.model_path)
    elif args.model_type == "lstm":
        batcher = Batcher(word2vec=word2vec, settings=parameters)

        if args.train:
            model = LSTM.Model(batcher, dataset, parameters)
            model.train()
        if args.test:
            model = LSTM.Model(batcher, dataset, parameters, load_model=args.model_path)
            model.test(args.test_split)


    end = time.time() - start
    m, s = divmod(end, 60)
    h, m = divmod(m, 60)
    print "%d:%02d:%02d" % (h, m, s)