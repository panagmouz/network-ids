#!/bin/zsh

python nids_train.py --preprocessing SMOTE --model vae --dataset NSL_KDD --isMulticlass --bs 64 --layers 1 --beta 1

python nids_train.py --preprocessing SMOTE --model dnn --dataset NSL_KDD --isMulticlass --bs 64 --layers 2
python nids_train.py --preprocessing SMOTE --model ae --dataset NSL_KDD --isMulticlass --bs 64 --layers 2
python nids_train.py --preprocessing SMOTE --model vae --dataset NSL_KDD --isMulticlass --bs 64 --layers 2 --beta 1

python nids_train.py --preprocessing SMOTE --model dnn --dataset UNSW_NB15 --isMulticlass --bs 64 --layers 1
python nids_train.py --preprocessing SMOTE --model ae --dataset UNSW_NB15 --isMulticlass --bs 64 --layers 1
python nids_train.py --preprocessing SMOTE --model vae --dataset UNSW_NB15 --isMulticlass --bs 64 --layers 1 --beta 1

python nids_train.py --preprocessing SMOTE --model dnn --dataset UNSW_NB15 --isMulticlass --bs 64 --layers 2
python nids_train.py --preprocessing SMOTE --model ae --dataset UNSW_NB15 --isMulticlass --bs 64 --layers 2
python nids_train.py --preprocessing SMOTE --model vae --dataset UNSW_NB15 --isMulticlass --bs 64 --layers 2 --beta 1