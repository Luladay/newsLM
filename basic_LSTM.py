import pickle
import tensorflow as tf
import numpy as np
import util
from decimal import Decimal
from datetime import datetime
import matplotlib.pyplot as py


today = datetime.now().strftime("%m-%d, %H:%M")

def generatePlots(x, y, xlabel, ylabel, title):	
	py.figure(figsize=(10,8))

	py.plot(x, y, color="blue")
	# py.plot(x, testAccReg, "b--", label='Test Accuracy (Lyrics)')
	# py.plot(x, trainAccAug, label='Train Accuracy (Lyrics + Audio)', color="red")
	# py.plot(x, testAccAug, linestyle="--", color="red", label='Test Accuracy (Lyrics + Audio)')
	# py.figlegend()
	py.title(title, fontsize="large")
	py.xlabel(xlabel, fontsize="large")
	py.ylabel(ylabel, fontsize="large")
	py.savefig("graphs/" + title + " " + today + ".png", bbox_inches="tight")

   
#since test() relies on default value of hidden_size and lr, be sure to update default value once it's tuned!!!!
def build_model(data_matrix, data_labels, hidden_size=256, lr=0.001):
	n_features = util.glove_dimensions
	n_classes = 5

	# add placeholders
	input_placeholder = tf.placeholder(tf.int32, shape=(None, util.short_article_len))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes))

	# add embedding layer!
	print "Opening embedding matrix..."	
	embed_matrix = util.openPkl("embeddings_matrix.pkl")
	print "Done opening embedding matrix!"
	x = tf.nn.embedding_lookup(embed_matrix, input_placeholder)

	# build model
	U = tf.get_variable("U", shape=[hidden_size, n_classes], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
	b = tf.get_variable("b", shape=[1, n_classes], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
    
	rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
	outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float64)

	h = final_state[1]
	pred = tf.matmul(h, U) + b

	loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred)
	loss_op = tf.reduce_mean(loss_op, 0)

	train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_op)
	return pred, input_placeholder, labels_placeholder, train_op, loss_op


def train(data_matrix, data_labels, save_path, title, hidden_size=256, lr=0.001, saved_model_path=None, RESUME=False, batch_size=256, n_epochs=30):
	tf.reset_default_graph()
	_, input_placeholder, labels_placeholder, train_op, loss_op = build_model(data_matrix, data_labels, hidden_size=hidden_size, lr=lr)	
	saver = tf.train.Saver()	
	avg_loss_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if RESUME:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, saved_model_path)
			print("Model restored.")

		minibatches = util.get_minibatches(data_matrix, data_labels, batch_size)
		for i in range(n_epochs):
			batch_loss_list = []
			print "Epoch " + str(i+1) + ": "
			for tup in minibatches:		
				_, loss = sess.run([train_op, loss_op], feed_dict={input_placeholder: tup[0], labels_placeholder: tup[1]})
				batch_loss_list.append(loss)
			avg_loss_list.append(np.mean(batch_loss_list))
			print "=====>loss: " + str(avg_loss_list[i]) + " "
			if (i > 0) and (avg_loss_list[i] < avg_loss_list[i-1]):
				tmp_path = save_path + "--smallest loss"
				saver.save(sess, tmp_path)
				print "New min loss at epoch %s! Model saved in path %s" % (str(i+1), tmp_path)
		saver.save(sess, save_path)
  		print("Final model saved in path: %s" % save_path)

  	util.dumpVar("losses/ " + title + " " + today + ".pkl" , avg_loss_list)
  	generatePlots(range(len(avg_loss_list)), avg_loss_list, "Number of Epochs", "Cross-Entropy Loss", title)


def test(data_matrix, data_labels, saved_model_path, title, batch_size=256):
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

	util.outputConfusionMatrix(pred_list, label_list, "confusion_matrix " + title + " " + today)
	util.get_accuracy(pred_list, label_list)


if __name__ == '__main__':

	print "Opening train data..."
	train_matrix = util.openPkl("train_matrix_rnn_short.pkl")
	train_labels = util.openPkl("train_labels_rnn_short.pkl")
	print "Done opening train data!"
	# print ">>>>Hidden size"
	# print "Running experiment 1..."
	# train(train_matrix, train_labels, "./models/basic_lstm_hsize256 lr01", "Basic LSTM hsize256 lr01", 
	# 	hidden_size=256, lr=0.001, saved_model_path="./models/basic_lstm_hsize256 lr01", RESUME=True, batch_size=256, n_epochs=5)
	# print "Running experiment 2..."
	# train(train_matrix, train_labels, "./models/basic_lstm_hsize300 lr01", "Basic LSTM hsize300 lr01", 
	# 	hidden_size=300, lr=0.001, RESUME=False, batch_size=256, n_epochs=45)
	# print "Running experiment 3..."
	# train(train_matrix, train_labels, "./models/basic_lstm_hsize512 lr01", "Basic LSTM hsize512 lr01", 
	# 	hidden_size=512, lr=0.001, RESUME=False, batch_size=256, n_epochs=45)
	print ">>>>Learning rate pt2"
	print "Running experiment 1..."
	train(train_matrix, train_labels, "./models/basic_lstm_hsize256 lr05", "Basic LSTM hsize256 lr05", 
		hidden_size=256, lr=0.005, RESUME=False, batch_size=256, n_epochs=30)
	print "Running experiment 2..."
	train(train_matrix, train_labels, "./models/basic_lstm_hsize300 lr05", "Basic LSTM hsize300 lr05", 
		hidden_size=300, lr=0.005, RESUME=False, batch_size=256, n_epochs=30)
	# print "Running experiment 3..."
	# train(train_matrix, train_labels, "./models/basic_lstm_hsize512 lr05", "Basic LSTM hsize512 lr05", 
	# 	hidden_size=512, lr=0.005, RESUME=False, batch_size=256, n_epochs=45)

	# print "Opening dev data..."
	# dev_matrix = util.openPkl("dev_matrix_rnn_short.pkl")	
	# dev_labels = util.openPkl("dev_labels_rnn_short.pkl")
	# print "Done opening dev data!"
	# print "------------"
	# print "Evaluating model on hsize256 lr01 +10"
	# test(dev_matrix, dev_labels, "./models/basic_lstm_hsize256 lr01--smallest loss", "Basic LSTM hsize256 lr01", batch_size=256)
	# print "Evaluating model on hsize300"
	# test(dev_matrix, dev_labels, "./models/basic_lstm_hsize300--smallest loss", "hsize300", batch_size=256)
	# print "Evaluating model on hsize512"
	# test(dev_matrix, dev_labels, "./models/basic_lstm_hsize512--smallest loss", "hsize512", batch_size=256)

