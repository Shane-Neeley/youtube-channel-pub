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
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class ANN(object):
    """
    run: python main.py --model ann --useall Y --epochs 150
    """
    def __init__(self, input_size, num_of_classes,
                 learning_rate=1E-6, hidden_layers=8, hidden_units=128,
                 optimizer='adam', loss='categorical_crossentropy',
                 activation='relu', dropout_p=0.5, batch_size=64, epochs=30):
        """
        Initialization for the Character Level CNN model.
        Args:
            --
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.num_of_classes = num_of_classes
        self.activation = activation
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.loss = loss
        self._build_model()  # builds self.model variable


    def _build_model(self):
        """
        Build and compile the model
        Returns: None
        """
        # the model will be a sequence of layers
        model = Sequential()
        # input layer
        model.add(Dense(units=self.hidden_units, input_dim=self.input_size, activation=self.activation))
        # hidden layers
        for layernumber in range(self.hidden_layers):
            model.add(Dense(units=self.hidden_units, activation=self.activation))
        # model.add(Dropout(self.dropout_p))
        # output layer
        model.add(Dense(self.num_of_classes, activation='softmax'))

        # Compile model
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        model.summary()
        self.model = model
        self.mname = 'ann'

    def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels,
              epochs, batch_size):
        """
        Training function
        Args:
            training_inputs (numpy.ndarray): Training set inputs
            training_labels (numpy.ndarray): Training set labels
            validation_inputs (numpy.ndarray): Validation set inputs
            validation_labels (numpy.ndarray): Validation set labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        Returns: None
        """

        # Start training
        print("Training model ANN ...")
        r = self.model.fit(
            training_inputs,
            training_labels,
            validation_data=(validation_inputs, validation_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2
        )

        # plot the loss and accuracies
        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.plot(r.history['acc'], label='acc')
        plt.plot(r.history['val_acc'], label='val_acc')
        plt.legend()
        plt.savefig('./logs/training_ann.png')
        plt.close()

        # Save model
        self.model.save('./logs/model_ann.h5', overwrite=True)


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
        possible_labels = np.genfromtxt("./data/classes.txt", dtype=np.str, delimiter='\n')
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
        # print(testing_inputs)
        print('testing_inputs.shape', testing_inputs.shape)
        y_pred = self.model.predict(testing_inputs)
        corrects = np.argmax(y_pred, axis=1) == np.argmax(testing_labels, axis=1)
        inc_ct = 0
        classcounts = {}
        for c in possible_labels:
            classcounts[c] = {"correct": 0, "incorrect": 0}
        with open('./logs/incorrects_ann.txt', 'w') as inc_f:
            # for index, boolean
            for idx, i in enumerate(corrects):
                # labels indexed at
                actual_class = possible_labels[ int(raw_test_data[idx][0])-1 ]
                predicted_class = possible_labels[np.argmax(y_pred[idx])]
                # corrects
                if i == True:
                    classcounts[actual_class]["correct"] += 1
                    pass
                # incorrects
                else:
                    classcounts[actual_class]["incorrect"] += 1
                    # msg = [raw_test_data[idx][1]] # need ID in raw
                    msg = []
                    msg.append("(" + actual_class + ")")
                    msg.append('predicted as:')
                    msg.append(predicted_class)
                    inc_f.write('\t'.join(msg) + '\n')
                    inc_ct += 1

        def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if not title:
                if normalize:
                    title = 'Normalized confusion matrix'
                else:
                    title = 'Confusion matrix, without normalization'

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            # Only use the labels that appear in the data
            classes = classes[unique_labels(y_true, y_pred)]
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title=title,
                   ylabel='True label',
                   xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            plt.savefig('./logs/confusion_matrix_' + self.mname + '.png')
            return ax


        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        # plot_confusion_matrix(
        #     np.argmax(testing_labels, axis=1),
        #     np.argmax(y_pred, axis=1),
        #     classes=possible_labels,
        #     title='Confusion matrix, without normalization'
        # )

        plot_confusion_matrix(
            np.argmax(testing_labels, axis=1),
            np.argmax(y_pred, axis=1),
            classes=possible_labels,
            normalize=True,
            title='Normalized confusion matrix'
        )

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
        plt.savefig('./logs/class_accuracies_ann.png')
        plt.close()

        print('\nincorrect/total:', inc_ct, ' / ', testing_inputs.shape[0], '\n')
