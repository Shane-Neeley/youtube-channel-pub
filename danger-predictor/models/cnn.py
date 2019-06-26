import numpy as np
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class CNN(object):
    """
    Class to implement the Character Level Convolutional Neural Network for Text Classification,
    as described in Zhang et al., 2015 (http://arxiv.org/abs/1509.01626)
    """
    def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers, num_of_classes,
                 threshold, dropout_p, learning_rate,
                 optimizer='adam', loss='categorical_crossentropy'):
        """
        Initialization for the Character Level CNN model.
        Args:
            input_size (int): Size of input features
            alphabet_size (int): Size of alphabets to create embeddings for
            embedding_size (int): Size of embeddings
            conv_layers (list[list[int]]): List of Convolution layers for model
            fully_connected_layers (list[list[int]]): List of Fully Connected layers for model
            num_of_classes (int): Number of classes in data
            threshold (float): Threshold for Thresholded ReLU activation function
            dropout_p (float): Dropout Probability
            optimizer (str): Training optimizer
            loss (str): Loss function
        """
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.num_of_classes = num_of_classes
        self.threshold = threshold
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.loss = loss
        self._build_model()  # builds self.model variable

    def _build_model(self):
        """
        Build and compile the Character Level CNN model
        Returns: None
        """
        # Input layer
        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(inputs)
        # Convolution layers
        for cl in self.conv_layers:
            x = Convolution1D(cl[0], cl[1])(x)
            x = ThresholdedReLU(self.threshold)(x)
            if cl[2] != -1:
                x = MaxPooling1D(cl[2])(x)
        x = Flatten()(x)
        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout_p)(x)
        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)
        # Build and compile model
        model = Model(inputs=inputs, outputs=predictions)

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )
        self.model = model
        print("CharCNNZhang model built: ")
        self.model.summary()

    def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels,
              epochs, batch_size, checkpoint_every=100):
        """
        Training function
        Args:
            training_inputs (numpy.ndarray): Training set inputs
            training_labels (numpy.ndarray): Training set labels
            validation_inputs (numpy.ndarray): Validation set inputs
            validation_labels (numpy.ndarray): Validation set labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_every (int): Interval for logging to Tensorboard
        Returns: None
        """
        # Create callbacks
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=checkpoint_every, batch_size=batch_size,
                                  write_graph=False, write_grads=True, write_images=False,
                                  embeddings_freq=checkpoint_every,
                                  embeddings_layer_names=None)
        # Start training
        print("Training CharCNNZhang model: ")
        r = self.model.fit(training_inputs, training_labels,
                       validation_data=(validation_inputs, validation_labels),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=2,
                       # callbacks=[tensorboard]
                       )

        # plot the loss and accuracies
        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.plot(r.history['acc'], label='acc')
        plt.plot(r.history['val_acc'], label='val_acc')
        plt.legend()
        plt.savefig('./logs/training.png')
        plt.close()
        # plt.show()

        # Save model
        self.model.save('./logs/model_zhang.h5', overwrite=True)


    def test(self, raw_test_data, testing_inputs, testing_labels, batch_size):
        """
        Testing function
        Args:
            testing_inputs (numpy.ndarray): Testing set inputs
            testing_labels (numpy.ndarray): Testing set labels
            batch_size (int): Batch size
        Returns: None
        """

        # classes file generated in jupyter notebook
        possible_labels = np.genfromtxt("./data/mm/classes.txt", dtype=np.str, delimiter='\n')
        print('classes:', possible_labels)

        # plot the mean AUC over each label
        print('Calculating Area Under Curve ...')
        p = self.model.predict(testing_inputs)
        aucs = []
        for j in range(len(possible_labels)):
            auc = roc_auc_score(testing_labels[:,j], p[:,j])
            aucs.append(auc)
        print('mean test AUC', np.mean(aucs), '\n')

        print('\nIncorrect Classifications')
        # indices of incorrect positions
        print(testing_inputs)
        print('testing_inputs.shape')
        print(testing_inputs.shape)
        y_pred = self.model.predict(testing_inputs)
        corrects = np.argmax(y_pred, axis=1) == np.argmax(testing_labels, axis=1)
        inc_ct = 0
        classcounts = {}
        for c in possible_labels:
            classcounts[c] = {"correct": 0, "incorrect": 0}
        with open('./logs/incorrects.txt', 'w') as inc_f:
            for idx, i in enumerate(corrects):
                actual_class = possible_labels[int(raw_test_data[idx][0])-1]
                predicted_class = possible_labels[np.argmax(y_pred[idx])]
                # corrects
                if i == True:
                    classcounts[actual_class]["correct"] += 1
                    pass
                # incorrects
                else:
                    classcounts[actual_class]["incorrect"] += 1
                    msg = [raw_test_data[idx][1]]
                    msg.append("(" + actual_class + ")")
                    msg.append('predicted as:')
                    msg.append(predicted_class)
                    inc_f.write('\t'.join(msg) + '\n')
                    inc_ct += 1

        x = []
        y = []
        for c in classcounts:
            total = classcounts[c]['correct'] + classcounts[c]['incorrect']
            percent = (classcounts[c]['correct'] / total) * 100
            x.append(c)
            y.append(percent)

        print(classcounts)
        print(x)
        print(y)

        # plot it
        fig, ax = plt.subplots()
        width = 0.75 # the width of the bars
        ind = np.arange(len(y))  # the x locations for the groups
        rects = ax.bar(ind, y, width)
        ax.set_xticks(ind)
        ax.set_xticklabels(x)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Percent')

        ##################
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width()/2.,
                    0.92*height,
                    str('%.3g'%(height)),
                    ha='center',
                    va='bottom'
                )
        ##################
        autolabel(rects)

        plt.title('Class Validation Accuracies')
        plt.tight_layout()
        plt.savefig('./logs/class_accuracies.png')
        plt.show()
        plt.close()

        print('\nincorrect/total:', inc_ct, ' / ', testing_inputs.shape[0], '\n')
