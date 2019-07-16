import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class KNeighbors(object):
    """
    """
    def __init__(self):
        """
        Returns: None
        """
        self._build_model()  # builds self.model variable

    def _build_model(self):
        """
        Returns: None
        """
        # Input
        self.model = KNeighborsClassifier()
        self.mname = 'kneighbors'

        # self.model = LogisticRegression()
        # self.mname = 'logreg'

        # self.model = DecisionTreeClassifier()
        # self.mname = 'decisiontree'

        # self.model = LinearDiscriminantAnalysis()
        # self.mname = 'lda'

        # self.model = GaussianNB()
        # self.mname = 'bayes'

        # self.model = SVC(gamma='auto')
        # self.mname = 'svc'

    def train(self, **kwargs):
        """
        Training function
        Args:
        Returns: None
        """
        X = kwargs['training_inputs']
        Y = kwargs['training_labels']
        print('X.shape', X.shape)
        print('Y.shape', Y.shape)
        print('fitting model')
        self.model = self.model.fit(X, Y)
        # Save model
        dump(self.model, './logs/model_' + self.mname + '.joblib')


    def test(self, **kwargs):
        """
        Testing function
        Args:
            testing_inputs (numpy.ndarray): Testing set inputs
            testing_labels (numpy.ndarray): Testing set labels
            batch_size (int): Batch size
        Returns: None
        """
        testing_inputs = kwargs['testing_inputs']
        testing_labels = kwargs['testing_labels']
        raw_test_data = kwargs['raw_test_data']

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
        with open('./logs/incorrects_' + self.mname + '.txt', 'w') as inc_f:
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
        plt.savefig('./logs/class_accuracies_' + self.mname + '.png')
        plt.close()

        print('\nincorrect/total:', inc_ct, ' / ', testing_inputs.shape[0], '\n')
