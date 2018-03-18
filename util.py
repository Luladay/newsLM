import pickle

# these imports are requiured for the confusion matrix
import numpy as np
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


UNK = '<UNK>'

nypost = "New_York_Post_articles"
bart = "Breitbart_articles"
cnn = "CNN_articles"
wpost = "Washington_Post_articles"
npr = "NPR_articles"

classes = ["New York Post", "Breitbart", "CNN", "Washington Post", "NPR"]

####### Variables #######

nypost_label = 0
bart_label = 1
cnn_label = 2
wpost_label = 3
npr_label = 4

short_article_len = 100
glove_dimensions = 50


######## Useful Functions #######

def dumpVar(filename, obj):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def openPkl(filename):
	var = open(filename, "rb")
	return pickle.load(var)

def get_minibatches(data_matrix, data_labels, batch_size):
	batch_list = []
	indices = []
	n_matrix_rows = data_matrix.shape[0] #dev or training examples
	for i in range(0, n_matrix_rows, batch_size):
		batch = data_matrix[i : i+batch_size, : ]
		batch_label = data_labels[i : i+batch_size]
		batch_list.append((batch, batch_label))
	return batch_list



def outputConfusionMatrix(pred, labels, filename):
    """ Generate a confusion matrix """

    cm = confusion_matrix(labels, pred, labels= range(5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
	# order of classes is consitent with create_vocab_embed_matrix.py line 31
    # classes = util.classes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)    

#this function can be used to test accuracy for train dev and test
def get_accuracy(pred, labels):
    """ Precision for classifier """
    prec = 2
    non_matching = 0
    for tup in zip(pred, labels):
    	if tup[0] != tup[1]:
    		non_matching += 1
    accuracy = non_matching / float(len(pred))
    print "Accuracy: " + str(round(accuracy * 100, prec)) + "%"

    micro_f1 = f1_score(labels, pred, average="micro")
    macro_f1 = f1_score(labels, pred, average="macro")
    class_f1 = f1_score(labels, pred, average=None)
    print "Micro F1 score: " + str(round(micro_f1 * 100, prec)) + "%"
    print "Macro F1 score: " + str(round(macro_f1 * 100, prec)) + "%"
    for class_name, score in zip(classes, class_f1):    	
    	print "F1 score for " + class_name + ": ", str(round(score*100, 3)) + "%"


####### Miscellaneous Functions	########
def checkForNans():
	print "Opening train matrix..."
	train_matrix = openPkl("train_matrix_short.pkl")
	print "Done opening test matrix!"
	for i, row in enumerate(train_matrix):
		if np.isnan(row).any():
			print "row num: ", i
			print "row values: ", row
			print " "

	print "Opening test_matrix matrix..."
	test_matrix = openPkl("test_matrix_short.pkl")
	print "Done opening test matrix!"
	for i, row in enumerate(test_matrix):
		if np.isnan(row).any():
			print "row num: ", i
			print "row values: ", row
			print " "

	print "Opening dev matrix..."
	dev_matrix = openPkl("dev_matrix_short.pkl")
	print "Done opening dev matrix!"
	for i, row in enumerate(dev_matrix):
		if np.isnan(row).any():
			print "row num: ", i
			print "row values: ", row
			print " "

def test_minibatches():
	data_matrix = np.arange(20).reshape((10,2))
	data_labels = np.arange(10)
	batch_size = 4
	print "data_matrix: ", data_matrix
	print ""
	print "data_labels: ", data_labels
	print ""
	batches = get_minibatches(data_matrix, data_labels, batch_size)
	for tup in batches:
		print tup

if __name__ == '__main__':
	a = [0,1,2,3,4]
	b = [1,1,2,4,1]
	get_accuracy(a,b)