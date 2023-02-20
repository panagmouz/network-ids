#!/usr/bin/python
import numpy as np
import argparse

from utils.preprocessing import basic
from datasets.NslKdd import NslKdd
from datasets.UnswNb15 import UnswNb15

parser = argparse.ArgumentParser(description='Network Intrusion Detection System (Network-IDS)')
parser.add_argument('--model', dest='model', default='vae', choices=['vae', 'ae'])
parser.add_argument('--layers', dest='layers', type=int, default=1, choices=[1, 2, 3, 4, 5])
parser.add_argument('--dataset', dest='dataset', default='NSL_KDD', choices=['NSL_KDD', 'UNSW_NB15'])
parser.add_argument('--testCase', dest='testCase', default='test-21', choices=['test', 'test-21'])
parser.add_argument('--isMulticlass', dest='isMulticlass', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--preprocessing', dest='preprocessing', default='basic', choices=['basic', 'random_sampling_SMOTE', 'SMOTE'])
parser.add_argument('--beta', dest='beta', type=float, default=0.001)
parser.add_argument('--seed', dest='seed', type=int, default=13)

args = parser.parse_args()
np.random.seed(args.seed)

dataset = None
if args.dataset == 'NSL_KDD':
    print("Dataset %s loading..." % args.dataset)
    dataset = NslKdd(testCase=args.testCase, isMulticlass=args.isMulticlass)
    dataset = dataset.get()
elif args.dataset == 'UNSW_NB15':
    print("Dataset %s loading..." % args.dataset)
    dataset = UnswNb15(testCase=args.testCase, isMulticlass=args.isMulticlass)
    dataset = dataset.get()
else:
    exit()

dataset_split, sizes = basic(data=dataset, isMulticlass=args.isMulticlass)
x_train, x_test, y_train, y_test = dataset_split
input_features, num_cls = sizes

checkpoint_root = '.results/%s_%s/%s/%s/%s/' % (args.dataset,
                                                args.testCase,
                                                'multiclass' if args.isMulticlass else 'binary',
                                                args.model + str(args.layers),
                                                args.preprocessing)

if args.model == 'vae':
    print("Model creation...")
    from models.vae import vae_ls, ls_extract
    model = vae_ls(input_features=input_features, layers=args.layers)
elif args.model == 'ae':
    print("Model creation...")
    from models.ae import ae_ls, ls_extract
    model = ae_ls(input_features=input_features, layers=args.layers)

print("Loading model...")
model.load_weights(checkpoint_root + "best-f1.hdf5", skip_mismatch=True, by_name=True)

print("Testing starts...")
ls_extract(args=args, model=model, data=(x_train, x_test, y_train, y_test))
