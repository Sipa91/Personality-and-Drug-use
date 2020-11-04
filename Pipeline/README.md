# ds-modeling-pipeline
This is a random forest model for predicting wether a person is likely to use drugs or not depending on demografics and certain personality traits.



##
Requirements:
- condamini or conda
- or pyenv with Python: 3.8.5

## Setup
Having Anaconda installed then create your ENV with

```bash
make setup-conda
```

With pyenv installed

```bash
make setup-pyenv
```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
python train.py  
```

In order to test that predict works on a test set you created run:

```bash
python predict.py models/rf_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible
