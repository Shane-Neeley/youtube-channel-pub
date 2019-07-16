from data_utils import Data
from data_utils import APIdata
from models.ann import ANN
from models.decisiontree import DecisionTree
from models.kneighbors import KNeighbors
import tensorflow as tf
import json
import sys
import os
import numpy as np
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
tf.flags.DEFINE_string('epochs', '30', 'epochs')
tf.flags.DEFINE_string('imports', 'N', 'import external data?')
tf.flags.DEFINE_string('useall', 'N', 'train with all data?')
FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
##########################################

if __name__ == "__main__":
    # Load configurations from file
    config = json.load(open('config.json'))
    FLAGS.epochs = int(FLAGS.epochs)
    # import data from external APIs
    if FLAGS.imports == "Y":
        APIdataclass = APIdata()
        dps = APIdataclass.inaturalistAPI()
        APIdataclass.weatherAPI(dps)

    classes = np.genfromtxt("data/classes.txt", dtype=np.str, delimiter='\n')

    #################################################
    # Load training data
    training_data = Data(
        data_source=config['data']['training_data_source'],
        num_of_classes=len(classes)
    )
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()

    # Load validation data for testing
    validation_data = Data(
        data_source=config['data']['validation_data_source'],
        num_of_classes=len(classes)
    )
    raw_test_data = validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()

    # Train with all data for a production model
    if FLAGS.useall == 'Y':
        training_inputs = np.concatenate((training_inputs, validation_inputs), axis=0)
        training_labels = np.concatenate((training_labels, validation_labels), axis=0)

    #################################################
    # Load model configurations and build model
    if FLAGS.model == 'cnn':
        model = CNN(
            input_size=training_inputs.shape[1],
            fully_connected_layers=config['cnn']['fully_connected_layers'],
            conv_layers=config['cnn']['conv_layers'],
            num_of_classes=len(classes)
        )
    elif FLAGS.model == 'kneighbors':
        model = KNeighbors()
    elif FLAGS.model == 'decisiontree':
        model = DecisionTree()
    # default ANN model
    else:
        model = ANN(
            input_size=training_inputs.shape[1],
            num_of_classes=len(classes)
        )

    #################################################
    # Train model
    model.train(
        training_inputs=training_inputs,
        training_labels=training_labels,
        validation_inputs=validation_inputs,
        validation_labels=validation_labels,
        epochs=FLAGS.epochs,
        batch_size=config['training']['batch_size']
    )

    #################################################
    # Test model
    model.test(
        raw_test_data=raw_test_data,
        testing_inputs=validation_inputs,
        testing_labels=validation_labels,
        batch_size=config['training']['batch_size']
    )
