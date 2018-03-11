import pickle
import tensorflow as tf
import numpy as np
import util

# these imports are requiured for the confusion matrix
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def outputConfusionMatrix(pred, labels, filename):
    """ Generate a confusion matrix """
    cm = confusion_matrix(labels, pred, labels=range(5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
	# order of classes is consitent with create_vocab_embed_matrix.py line 31
    classes = ["New York Post", "Breitbart", "CNN", "Washington Post", "NPR"]
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

#this functionc an be used to test accuracy for train dev and test
def accuracy(label, pred):
    """ Precision for classifier """
    assert(label.shape == pred.shape)
    return np.sum(label == pred) * 100.0 / pred.size



def run_baseline(data_matrix, data_labels, train=True):
	n_features = util.glove_dimensions
	n_classes = 5
	batch_size = 1000
	n_epochs = 30
	lr = 0.005
	pred_list = [] # will return the list of predictions for confusion matrix

	input_placeholder = tf.placeholder(tf.float32, shape=(None, n_features))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes))

	W = tf.Variable(tf.zeros([n_features, n_classes], dtype=tf.float32))
	b = tf.Variable(tf.zeros([n_classes, 1], dtype=tf.float32))

	xW = tf.matmul(input_placeholder, W)
	pred = tf.transpose(tf.transpose(xW) + b)
	loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred)
	loss_op = tf.reduce_mean(loss_op, 0)

	train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_op)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		minibatches = get_minibatches(data_matrix, data_labels, batch_size)
		for i in range(n_epochs):




			print "Epoch " + str(i+1) + ": "
			loss_list = []
			for ii, tup in enumerate(minibatches):

				if train:
					_, loss = sess.run([train_op, loss_op], feed_dict={input_placeholder: tup[0], labels_placeholder: tup[1]})
				else:
						# store the predictions from the last epoch run
					if (i == n_epochs - 1):
						if (ii == 0):
							pred_list = tup[0]
						else:
							np.stack(pred_list, tup[0])

					loss = sess.run(loss_op, feed_dict={input_placeholder: tup[0], labels_placeholder: tup[1]})
				loss_list.append(loss)
			print "=====>loss: " + str(np.mean(loss_list)) + " "
			if not train:
				return pred_list
				break


def get_minibatches(data_matrix, data_labels, batch_size):
	batch_list = []
	indices = []
	n_matrix_rows = data_matrix.shape[0] #dev or training examples
	for i in range(0, n_matrix_rows, batch_size):
		batch = data_matrix[i : i+batch_size, : ]
		batch_label = data_labels[i : i+batch_size]
		batch_list.append((batch, batch_label))
	return batch_list

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

def checkForNans():
	print "Opening train matrix..."
	train_matrix = util.openPkl("train_matrix_short.pkl")
	print "Done opening test matrix!"
	for i, row in enumerate(train_matrix):
		if np.isnan(row).any():
			print "row num: ", i
			print "row values: ", row
			print " "

	print "Opening test_matrix matrix..."
	test_matrix = util.openPkl("test_matrix_short.pkl")
	print "Done opening test matrix!"
	for i, row in enumerate(test_matrix):
		if np.isnan(row).any():
			print "row num: ", i
			print "row values: ", row
			print " "

	print "Opening dev matrix..."
	dev_matrix = util.openPkl("dev_matrix_short.pkl")
	print "Done opening dev matrix!"
	for i, row in enumerate(dev_matrix):
		if np.isnan(row).any():
			print "row num: ", i
			print "row values: ", row
			print " "


if __name__ == '__main__':
	# change these filenames if the pickle files are in a Data folder

	# print "Opening train data..."
	# train_matrix = util.openPkl("train_matrix_short.pkl")
	# train_labels = util.openPkl("train_labels_short.pkl")
	# print "Done opening training data!"
	# run_baseline(train_matrix, train_labels)


	# print "Opening test data..."
	# test_matrix = util.openPkl("test_matrix_short.pkl")
	# test_labels = util.openPkl("test_labels_short.pkl")
	# print "Done opening test data!"
	# run_baseline(test_matrix, test_labels, train=False)


	print "Opening dev data..."
	dev_matrix = util.openPkl("dev_matrix_short.pkl")
	dev_labels = util.openPkl("dev_labels_short.pkl")
	print "Done opening dev data!"
	pred_list = run_baseline(dev_matrix, dev_labels, train=False)
	outputConfusionMatrix(pred_list, dev_labels, confusion_matrix)
