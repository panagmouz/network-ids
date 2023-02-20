## Description
In this project 3 Deep Learning models are developed that can be used in Network Intrusion Detection Systems (NIDS). This project was part of my MSC Diploma thesis.

## Folder structure
The folder ./datasets contains the data loaders for each dataset that has been used.

The folder ./models contains all models developed during this project. Each file in this folder describes the model architecture, the train, and the test functions.

The folder ./utils contains scripts that describe callback, loss, and preprocessing functions.

## How to run
To run the python scripts in this repository you need to install and activate a python virtual environment. Also, you need to install all python libraries described in requirements.txt. So you have to run the following commands in your shell:

```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

In order to reproduce the train results you have to run:

```
./train_all.sh
```

The training results are autosaved in .results folder.
