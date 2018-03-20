import pickle
import tensorflow as tf
import numpy as np
import util
from decimal import Decimal

   
def build_model(data_matrix, data_labels):
	n_features = util.glove_dimensions
	n_classes = 5
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
	return pred, input_placeholder, labels_placeholder, train_op, loss_op


def train(data_matrix, data_labels, save_path, batch_size=100, n_epochs=30):
	_, input_placeholder, labels_placeholder, train_op, loss_op = build_model(data_matrix, data_labels)	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		minibatches = util.get_minibatches(data_matrix, data_labels, batch_size)
		for i in range(n_epochs):
			print "Epoch " + str(i+1) + ": "
			loss_list = []
			label_list= []
			pred_list = []
			for tup in minibatches:				
				_, loss = sess.run([train_op, loss_op], feed_dict={input_placeholder: tup[0], labels_placeholder: tup[1]})
				loss_list.append(loss)
			print "=====>loss: " + str(np.mean(loss_list)) + " "		
		save_path = saver.save(sess, save_path)
  		print("Final baseline model saved in path: %s" % save_path)


def test(data_matrix, data_labels, saved_model_path, batch_size=1000):
	tf.reset_default_graph()
	pred, input_placeholder, labels_placeholder, _, loss_op = build_model(data_matrix, data_labels)
	saver = tf.train.Saver()
	loss_list = []
	label_list= []
	pred_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, saved_model_path)
		print("Model restored.")

		minibatches = util.get_minibatches(data_matrix, data_labels, batch_size)
		for tup in minibatches:
			pred_temp, loss, labels_temp = sess.run([pred, loss_op, labels_placeholder], feed_dict={input_placeholder: tup[0], labels_placeholder: tup[1]})
			for i, row in enumerate(pred_temp):
				pred_list.append(np.where(row == max(row))[0][0])
			for i, row in enumerate(labels_temp):
				label_list.append(np.where(row == max(row))[0][0])

			loss_list.append(loss)
		print "Loss: " + str(np.mean(loss_list)) + "\n"			

	util.outputConfusionMatrix(pred_list, label_list, "confusion_matrix_baseline_test")
	util.get_accuracy(pred_list, label_list)



if __name__ == '__main__':
	# change these filenames if the pickle files are in a Data folder

	# print "Opening train data..."
	# train_matrix = util.openPkl("train_matrix_short.pkl")
	# train_labels = util.openPkl("train_labels_short.pkl")
	# print "Done opening train data!"
	# train(train_matrix, train_labels, "./models/baseline", batch_size=1000, n_epochs=1500)

	print "Opening test data..."
	dev_matrix = util.openPkl("test_matrix_short.pkl")	
	dev_labels = util.openPkl("test_labels_short.pkl")
	print "Done opening test data!"
	test(dev_matrix, dev_labels, "./models/baseline", batch_size=1000)
