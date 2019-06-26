from data_utils import Data
from data_utils import APIdata
from keras.models import load_model
import tensorflow as tf
import json
import sys
import os
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from joblib import dump, load
# https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#### Delete all flags before declare #####
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)
tf.flags.DEFINE_string('model', 'ann', 'specifies which model to use')
FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
##########################################

if __name__ == "__main__":

    #################################################
    # Load model configurations and build model
    if FLAGS.model == 'cnn':
        model = load_model('./logs/model_cnn.h5')
    elif FLAGS.model == 'kneighbors':
        model = load('./logs/model_kneighbors.joblib')
    elif FLAGS.model == 'decisiontree':
        model = load('./logs/model_decisiontree.joblib')
    else:
        model = load_model('./logs/model_ann.h5')

    #################################################

    # Load configurations from file
    config = json.load(open('config.json'))
    classes = np.genfromtxt("data/classes.txt", dtype=np.str, delimiter='\n')
    headers = np.genfromtxt("data/headers.txt", dtype=np.str, delimiter='\n')

    #################################################
    # Process data for today
    # need to call weather API for today .. and forecast?
    # build datapoints from today and forecast for preferred locations where you don't want to get attacked
    thousandAcres = (45.548664, -122.372049)
    datapoints = {}
    # today at noon
    du = dt.today()
    date_o = dt(du.year, du.month, du.day, 12)

    # today, 3 days and 7 days from now
    daysahead = [0,3,7]
    for days in daysahead:
        date_o2 = date_o + timedelta(days)
        datapoint = {
            'id': str(days),
            'time_observed_at': date_o2.isoformat(),
            'lat': thousandAcres[0],
            'lon': thousandAcres[1]
        }
        datapoints[str(days)] = datapoint

    d = APIdata().weatherAPI(datapoints, predict=True)
    data = []
    for dp in d:
        row = []
        # first header is class number
        for h in headers[1:]:
            row.append(d[dp][h])
        data.append(row)
    data = np.array(data, dtype='float64')
    predictions = model.predict(data)

    # today, 3 days and 7 days from now
    for num, d_pred in enumerate(predictions):
        daystr = ''
        if num == 0:
            daystr = 'today'
        else:
            daystr = str(daysahead[num]) + ' days from now'
        max_prob = np.argmax(d_pred)
        predicted_class = classes[max_prob]
        if d_pred[max_prob] > 0.5:
            print('\n', daystr, '\nHIGH CHANCE FOR:', predicted_class, d_pred[max_prob])
            print(d_pred, '\n')
        else:
            print('\n', daystr, '\nDID NOT MEET THRESHOLD:', predicted_class, d_pred[max_prob])
            print(d_pred, '\n')



    ########
