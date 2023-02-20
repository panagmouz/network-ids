#!/usr/bin/python
import numpy as np
from numpy import savetxt
import argparse

from utils.preprocessing import basic, random_sampling_SMOTE, mySMOTE
from datasets.Kdd import Kdd
from datasets.NslKdd import NslKdd
from datasets.UnswNb15 import UnswNb15

parser = argparse.ArgumentParser(description='Network Intrusion Detection System (Network-IDS)')
parser.add_argument('--model', dest='model', default='dnn', choices=['dnn', 'vae', 'ae'])
parser.add_argument('--layers', dest='layers', type=int, default=1, choices=[1, 2, 3, 4, 5])
parser.add_argument('--dataset', dest='dataset', default='NSL_KDD', choices=['KDD', 'NSL_KDD', 'UNSW_NB15'])
parser.add_argument('--testCase', dest='testCase', default='test', choices=['test', 'test-21'])
parser.add_argument('--isMulticlass', dest='isMulticlass', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--preprocessing', dest='preprocessing', default='basic', choices=['basic', 'random_sampling_SMOTE', 'SMOTE'])
parser.add_argument('--bs', dest='bs', type=int, default=64)
parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
parser.add_argument('--beta', dest='beta', type=float, default=0.001)
parser.add_argument('--epochs', dest='epochs', type=int, default=500)
parser.add_argument('--seed', dest='seed', type=int, default=13)
parser.add_argument('--pretrained', dest='pretrained', default='')

args = parser.parse_args()
np.random.seed(args.seed)

dataset = None
if args.dataset == 'NSL_KDD':
    print("Dataset %s loading..." % args.dataset)
    dataset = NslKdd(testCase=args.testCase, isMulticlass=args.isMulticlass)
    dataset = dataset.get()
elif args.dataset == 'KDD':
    print("Dataset %s loading..." % args.dataset)
    dataset = Kdd(testCase=args.testCase, isMulticlass=args.isMulticlass)
    dataset = dataset.get()
elif args.dataset == 'UNSW_NB15':
    print("Dataset %s loading..." % args.dataset)
    dataset = UnswNb15(testCase=args.testCase, isMulticlass=args.isMulticlass)
    dataset = dataset.get()
else:
    exit()

dataset_split = None
sizes = None
if args.preprocessing == 'basic':
    dataset_split, sizes = basic(data=dataset, isMulticlass=args.isMulticlass)
elif args.preprocessing == 'random_sampling_SMOTE':
    dataset_split, sizes = random_sampling_SMOTE(data=dataset, args=args, isMulticlass=args.isMulticlass)
elif args.preprocessing == 'SMOTE':
    dataset_split, sizes = mySMOTE(data=dataset, args=args, isMulticlass=args.isMulticlass)
else:
    exit()
x_train, x_test, y_train, y_test = dataset_split
input_features, num_cls = sizes

checkpoint_root = '.checkpoint_results_last2/%s_%s/%s/%s/%s/' % (args.dataset, args.testCase,
                                                           'multiclass' if args.isMulticlass else 'binary',
                                                           args.model + str(args.layers),
                                                           args.preprocessing)

if args.model == 'dnn':
    print("Model creation...")
    from models.dnn import dnn, train, test
    model = dnn(input_features=input_features, isMulticlass=args.isMulticlass, layers=args.layers, num_cls=num_cls)
elif args.model == 'vae':
    print("Model creation...")
    from models.vae import vae, train, test
    model = vae(input_features=input_features, isMulticlass=args.isMulticlass, layers=args.layers, num_cls=num_cls)
elif args.model == 'ae':
    print("Model creation...")
    from models.ae import ae, train, test
    model = ae(input_features=input_features, isMulticlass=args.isMulticlass, layers=args.layers, num_cls=num_cls)

if args.pretrained != '':
    print("Loading pretrained model...")
    model.load_weights(args.pretrained)

print("Training starts...")
train(args=args, model=model, checkpoint_root=checkpoint_root, data=(x_train, x_test, y_train, y_test))
model.save("%s/last.hdf5" % checkpoint_root)

print("Loading model...")
model.load_weights(checkpoint_root + "best-f1.hdf5")

print("Testing starts...")
conf_mat = test(args=args, model=model, data=(x_train, x_test, y_train, y_test))
savetxt(checkpoint_root + 'results.csv', conf_mat, delimiter=',', fmt='%d')
