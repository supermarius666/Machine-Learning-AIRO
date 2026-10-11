# Import libraries that contains the implementations of the functions used in the rest of the program.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np
import random

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
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

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

    ax.set_ylim(len(classes)-0.5, -0.5)

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
    return ax

# Load training data
iris = datasets.load_iris()
dataset_name = "Iris" # dataset name 
X_all = iris.data # features: sepal length, sepal width, petal length, petal width 
y_all = iris.target # classes: Setosa, Versicolor, Virginica
class_names = np.array([str(c) for c in iris.target_names])

# Print some information about the dataset
print("-------------------------------Info-----------------------------------")
print("Dataset: %s" %(dataset_name))
print("Input shape: %s" %str(X_all.shape)) # 150 examples with 4 features each (150, 4) matrix 
print("Output shape: %s" %str(y_all.shape)) # 150 outputs (150, 1) vector of dim 1
print("Number of attributes/features: %d" %(X_all.shape[1]))
print("Number of classes: %d %s" %(len(class_names), str(class_names)))
print("Number of samples: %d" %(X_all.shape[0]))
print("----------------------------------------------------------------------")

# Display a random sample
id = random.randrange(0,X_all.shape[0]) # random data point
print("----------------------------Random data point ------------------------")
print("x%d = %r" %(id,X_all[id]))
print("y%d = %r ['%s']" %(id,y_all[id],class_names[y_all[id]]))
print("----------------------------------------------------------------------")

print("--------------------------- Split the data ---------------------------")
# test_size(1/3): denote percentage of samples in the test set
# random_state: parameter is used as random seed
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.333, random_state=42) 

print("Size of training set: %d" %X_train.shape[0])
print("Size of test set: %d" %X_test.shape[0])

print('First training sample')
id = 0
print("    x_train_%d = %r" %(id,X_train[id]))
print("    y_train_%d = %r ['%s']" %(id,y_train[id],class_names[y_train[id]]))

print('First test sample')
id = 0
print("    x_test_%d = %r" %(id,X_test[id]))
print("    y_test_%d = %r ['%s']" %(id,y_test[id],class_names[y_test[id]]))
print("----------------------------------------------------------------------")

# Model 
model = tree.DecisionTreeClassifier()

# Dataset's standardization (not neede for DT)
use_standardization = True
if use_standardization:
  pipe = make_pipeline(preprocessing.StandardScaler(), model)
else:
  pipe = model

# Finding a solution for this model given the data set. The solution is stored in the model object.
pipe.fit(X_train, y_train)

# plot the tree
tree.plot_tree(model,fontsize=12)
plt.rcParams["figure.figsize"] = (30,10)
plt.show()

# Predict on test set
y_pred = pipe.predict(X_test)

# Evaluate the Model
# accuracy
print("-------------------------- Accuracy-----------------------------------")
acc = pipe.score(X_test, y_test)
print("Accuracy %.3f" %acc)
print("----------------------------------------------------------------------")

# precision and recall
print("-------------------- Precision and Recall ----------------------------")
print(classification_report(y_test, y_pred, labels=None, target_names=class_names, digits=3))
print("----------------------------------------------------------------------")

# confusion matrix
print("---------------------- Confusion Matrix-------------------------------")
cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
print(cm)
plt.rcParams["figure.figsize"] = (10,10)
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=False)
print("----------------------------------------------------------------------")


