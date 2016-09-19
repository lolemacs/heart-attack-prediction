##Requirements

- Python = 2.7

Python dependencies - depend hugely on which models will be used:
- Numpy
- Sklearn
- Xgboost
- Keras
- Theano

The easiest way to install them is via pip. Pip can be installed on Ubuntu with the default package manager:

`sudo apt-get install python-pip`

The dependencies can be further installed with:

`sudo pip install numpy sklearn xgboost theano keras`

##Description

The project is composed of several files, most of them being scripts to train and evaluate ML models.

The only different script is xlsx_to_pkl.py, which receives two arguments: the name of the xlsx file to load data from, and the name of the pkl file to dump the data onto. All models use .pkl files to load their data, so this step is necessary.

##Instructions

Each model can be used directly, without need to define command-line arguments.

Example:

`python logreg.py`

Each model script will print their respective F1 scores.
