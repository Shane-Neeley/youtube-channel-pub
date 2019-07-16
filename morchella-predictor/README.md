## Morchella (Genus of morel mushroom) & More! Occurrence Prediction App

Predict whether today is the day to go mushroom hunting! (in a more complex way than just picking up a guide, or asking an old Russian lady / Mexican uncle).

Gather historical weather data of mushroom sightings posted to iNaturalist.

Integrates w/ DarkSky weather API and iNaturalist API.

Get data points like geolocation, date-time, vegetative index, temperature, barometric pressure, past precipitation, precipitation probabilities.

Feed data into machine learning models. Neural networks (regular and convolutional), decision trees, k nearest neighbors algorithms. Choose the best performing model to train.

Run TODAY'S data (weather, geo, etc.) through the trained model to predict your chances of seeing which mushroom species!

Current observation genus: Morchella, Laetiporus, Cantharellus, Helvella (false morel), Verpa (false morel), Gyromitra, (false morel), Boletus, Pleurotus, Agaricus, Gyroporus, Leccinum.

## Setup and Run

#### Create files at project level:

- secret_key1.txt -- just one line w/ WEATHER_KEY from https://darksky.net/dev (credit card required, but won't cost you much.)
- secret_key2.txt -- just one line w/ AGRO_KEY from http://api.agromonitoring.com (*optional* and not working anyways)

#### To run:

Imports + training. Imports cost money depending on how many weather observations done.

`python main.py --imports Y`

Just training w/o imports.

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
