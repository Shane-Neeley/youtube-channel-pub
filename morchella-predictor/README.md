#### Create files at project level:

- secret_key1.txt -- just one line w/ WEATHER_KEY from https://darksky.net/dev (credit card required)
- secret_key2.txt -- just one line w/ AGRO_KEY from http://api.agromonitoring.com (*optional* and not working anyways)

#### To run:

Imports + training
`python main.py --imports Y`

Just training
`python main.py`

Train using all data available for a production model (otherwise uses TRAINING_SPLIT variable)
`python main.py --useall Y`

#### For daily predictions using trained models

ANN model
`python predict.py --model ann`

CNN model
`python predict.py --model cnn`

Decision tree model
`python predict.py --model decisiontree`

K-nearest neighbors model
`python predict.py --model kneighbors`

#### Mushroom recipes

Batter and Fry.
