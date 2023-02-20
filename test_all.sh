#!/bin/zsh


python nids_test.py --dataset NSL_KDD --isMulticlass --layers 1
python nids_test.py --dataset NSL_KDD --isMulticlass --layers 1 --testCase test-21
python nids_test.py --dataset KDD --isMulticlass --layers 1
python nids_test.py --dataset UNSW_NB15 --isMulticlass --layers 1

python nids_test.py --dataset NSL_KDD --isMulticlass --layers 2
python nids_test.py --dataset NSL_KDD --isMulticlass --layers 2 --testCase test-21
python nids_test.py --dataset KDD --isMulticlass --layers 2