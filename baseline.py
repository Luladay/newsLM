import pickle
import tensorflow as tf
import numpy as np
import util
from decimal import Decimal

# these imports are requiured for the confusion matrix
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def outputConfusionMatrix(pred, labels, filename):
    """ Generate a confusion matrix """

    # print('size of pred')
    # print(np.shape(pred))
    # print( "size of labels")
    # print(np.shape(labels))
    cm = confusion_matrix(labels, pred, labels= range(5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
	# order of classes is consitent with create_vocab_embed_matrix.py line 31
    classes = util.classes
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
def accuracy(pred, labels):
    """ Precision for classifier """
    prec = 2
    micro_f1 = f1_score(labels, pred, average="micro")
    macro_f1 = f1_score(labels, pred, average="macro")
    class_f1 = f1_score(labels, pred, average=None)
    print "Micro F1 score: " + str(round(micro_f1 * 100, prec)) + "%"
    print "Macro F1 score: " + str(round(macro_f1 * 100, prec)) + "%"
    for class_name, score in zip(util.classes, class_f1):    	
    	print "F1 score for " + class_name + ": ", str(round(score*100, 3)) + "%"
   

def run_baseline(data_matrix, data_labels, train=True):
	n_features = util.glove_dimensions
	n_classes = 5
	batch_size = 1000
	n_epochs = 30
	lr = 0.005

	input_placeholder = tf.placeholder(tf.float32, shape=(None, n_features))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes))


	W = tf.get_variable("W", shape =[n_features, n_classes], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	b = tf.Variable(tf.zeros([n_classes, 1], dtype=tf.float32))

	xW = tf.matmul(input_placeholder, W)
	pred = tf.transpose(tf.transpose(xW) + b)
	loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred)
	loss_op = tf.reduce_mean(loss_op, 0)

	train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_op)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		minibatches = util.get_minibatches(data_matrix, data_labels, batch_size)
		for i in range(n_epochs):
			if not train:
				print "Epoch " + str(i+1) + ": "
			loss_list = []
			label_list= []
			pred_list = []
			for tup in minibatches:
				if train:
					_, loss = sess.run([train_op, loss_op], feed_dict={input_placeholder: tup[0], labels_placeholder: tup[1]})
				else:
					pred_temp, loss, labels_temp = sess.run([pred, loss_op, labels_placeholder], feed_dict={input_placeholder: tup[0], labels_placeholder: tup[1]})
					for i, row in enumerate(pred_temp):
						pred_list.append(np.where(row == max(row))[0][0])
					for i, row in enumerate(labels_temp):
						label_list.append(np.where(row == max(row))[0][0])

				loss_list.append(loss)
			print "=====>loss: " + str(np.mean(loss_list)) + " "
			if not train:
				break
	return pred_list, label_list





if __name__ == '__main__':
	# change these filenames if the pickle files are in a Data folder

	 #print "Opening train data..."
	 #train_matrix = util.openPkl("train_matrix_short.pkl")
	 #train_labels = util.openPkl("train_labels_short.pkl")
	 #print "Done opening training data!"
	 #run_baseline(train_matrix, train_labels)


	# print "Opening test data..."
	# test_matrix = util.openPkl("test_matrix_short.pkl")
	# test_labels = util.openPkl("test_labels_short.pkl")
	# print "Done opening test data!"
	# run_baseline(test_matrix, test_labels, train=False)


	print "Opening dev data..."
	dev_matrix = util.openPkl("dev_matrix_short.pkl")
	dev_labels = util.openPkl("dev_labels_short.pkl")
	print "Done opening dev data!"
	pred_list, label_list = run_baseline(dev_matrix, dev_labels, train=False)
	outputConfusionMatrix(pred_list, label_list, "confusion_matrix")
	accuracy(pred_list, label_list)