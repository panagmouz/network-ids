#!/usr/bin/python
import argparse
import numpy as np
from numpy import savetxt

from utils.preprocessing import basic, undersampling_to_10k

parser = argparse.ArgumentParser(description='Network Intrusion Detection System (Network-IDS)')
parser.add_argument('--model', dest='model', default='dnn', choices=['dnn'])
parser.add_argument('--layers', dest='layers', type=int, default=1, choices=[1, 2, 3, 4, 5])
parser.add_argument('--dataset', dest='dataset', default='NSL_KDD', choices=['KDD', 'NSL_KDD', 'UNSW_NB15'])
parser.add_argument('--testCase', dest='testCase', default='test', choices=['test', 'test-21'])
parser.add_argument('--isMulticlass', dest='isMulticlass', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--preprocessing', dest='preprocessing', default='basic', choices=['basic'])
parser.add_argument('--bs', dest='bs', type=int, default=64)
parser.add_argument('--seed', dest='seed', type=int, default=13)

args = parser.parse_args()
np.random.seed(args.seed)

dataset = None
if args.dataset == 'NSL_KDD':
    print("Dataset %s loading..." % args.dataset)
    from datasets.NslKdd import NslKdd
    dataset = NslKdd(testCase=args.testCase, isMulticlass=args.isMulticlass)
    dataset = dataset.get()
elif args.dataset == 'KDD':
    print("Dataset %s loading..." % args.dataset)
    from datasets.Kdd import Kdd
    dataset = Kdd(testCase=args.testCase, isMulticlass=args.isMulticlass)
    dataset = dataset.get()
elif args.dataset == 'UNSW_NB15':
    print("Dataset %s loading..." % args.dataset)
    from datasets.UnswNb15 import UnswNb15
    dataset = UnswNb15(testCase=args.testCase, isMulticlass=args.isMulticlass)
    dataset = dataset.get()
else:
    exit()


dataset_split = None
sizes = None
if args.preprocessing == 'basic':
    dataset_split, sizes = basic(data=dataset, isMulticlass=args.isMulticlass)
elif args.preprocessing == 'undersampling':
    dataset_split, sizes = undersampling_to_10k(data=dataset, args=args, isMulticlass=args.isMulticlass)
else:
    exit()
x_train, x_test, y_train, y_test = dataset_split
input_features, num_cls = sizes

checkpoint_root = '.checkpoint_results/%s_%s/%s/%s/%s/' % (args.dataset, args.testCase,
                                                           'multiclass' if args.isMulticlass else 'binary',
                                                           args.model + str(args.layers),
                                                           args.preprocessing)

if args.model == 'dnn':
    print("Model creation...")
    from models.dnn import dnn, train, test
    model = dnn(input_features=input_features, isMulticlass=args.isMulticlass, layers=args.layers, num_cls=num_cls)

    print("Loading model...")
    model.load_weights(checkpoint_root + "best-f1.hdf5")

    print("Testing starts...")
    conf_mat = test(args=args, model=model, data=(x_train, x_test, y_train, y_test))
    savetxt(checkpoint_root + 'results.csv', conf_mat, delimiter=',', fmt='%d')
