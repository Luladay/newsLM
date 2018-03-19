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
def build_model(data_matrix, hidden_size=256, lr=0.001):
	n_features = util.glove_dimensions
	n_classes = util.vocab_size #7k
	# lr = 0.001
	# hidden_size = 256

	# add placeholders
	input_placeholder = tf.placeholder(tf.int32, shape=(None, util.short_article_len - 1))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes))

	# add embedding layer!
	print "Opening embedding matrix..."
	embed_matrix = util.openPkl("embeddings_matrix.pkl")
	print "Done opening embedding matrix!"
	x = tf.nn.embedding_lookup(embed_matrix, input_placeholder)

	# build model
	U = tf.get_variable("U", shape=[hidden_size, n_classes], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", shape=[1, n_classes], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
	b1 = tf.get_variable("b1", shape=[1, hidden_size], dtype=tf.float64, initializer=tf.constant_initializer(0.0))

	Wh = tf.get_variable("Wh", shape = [ hidden_size, hidden_size], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
	#We = tf.get_variable("We", shape = [hidden_size, n_features], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
	rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
	outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float64)

	h = tf.sigmoid( tf.matmul(Wh, final_state[1])  + b1)
	pred = tf.matmul(h, U) + b2

	loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred)
	loss_op = tf.reduce_mean(loss_op, 0)

	train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_op)
	return pred, input_placeholder, labels_placeholder, train_op, loss_op


def train(data_matrix, save_path, title, hidden_size=256, lr=0.001, saved_model_path=None, RESUME=False, batch_size=256, n_epochs=30):
	if RESUME:
		tf.reset_default_graph()
	_, input_placeholder, labels_placeholder, train_op, loss_op = build_model(data_matrix,  hidden_size=hidden_size, lr=lr)
	saver = tf.train.Saver()
	avg_loss_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if RESUME:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, saved_model_path)
			print("Model restored.")

		minibatches = util.get_minibatches_lm(data_matrix, batch_size)
		for i in range(n_epochs):
			batch_loss_list = []
			print "Epoch " + str(i+1) + ": "
			for tup in minibatches:
				label_data = np.zeros((len(tup[1]), util.vocab_size))
				label_data[np.arange(len(tup[1])), tup[1]] = 1

				_, loss = sess.run([train_op, loss_op], feed_dict={input_placeholder: tup[0], labels_placeholder: label_data})
				batch_loss_list.append(loss)
			avg_loss_list.append(np.mean(batch_loss_list))
			print "=====>loss: " + str(avg_loss_list[i]) + " "
			if (i > 0) and (avg_loss_list[i] < avg_loss_list[i-1]):
				tmp_path = save_path + "--smallest loss"
				saver.save(sess, tmp_path)
				print "New min loss at epoch %s! Model saved in path %s" % (str(i+1), tmp_path)
		saver.save(sess, save_path)
  		print("Final model saved in path: %s" % save_path)

  	generatePlots(range(len(avg_loss_list)), avg_loss_list, "Number of Epochs", "Cross-Entropy Loss", title)
  	util.dumpVar("losses/ " + title + " " + today + ".pkl" , avg_loss_list)


def test(data_matrix, saved_model_path, title):
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

		minibatches = util.get_minibatches_lm(data_matrix,  batch_size)
		for tup in minibatches:
			label_data = np.zeros((batch_size, util.vocab_size))
			label_data[np.arange(batch_size), tup[1]] = 1

			pred_temp, loss, labels_temp = sess.run([pred, loss_op, labels_placeholder], feed_dict={input_placeholder: tup[0], labels_placeholder: label_data})
			for i, row in enumerate(pred_temp):
				pred_list.append(np.where(row == max(row))[0][0])
			for i, row in enumerate(labels_temp):
				label_list.append(np.where(row == max(row))[0][0])

			loss_list.append(loss)
		print "Loss: " + str(np.mean(loss_list)) + "\n"

	#util.outputConfusionMatrix(pred_list, label_list, "confusion_matrix " + title + " " + today)
	util.get_accuracy(pred_list, label_list)


if __name__ == '__main__':

	print "Opening train data..."
	train_matrix = util.openPkl("train_matrix_rnn_short.pkl")
	#train_labels = util.openPkl("train_lm_labels.pkl")
	print "Done opening train data!"
	print ">>>>Hidden size"
	print "Running experiment 1..."
	train(train_matrix, "./models/lm_lstm_hsize256 lr01", "LM LSTM hsize256 lr01",
		hidden_size=256, lr=0.001, RESUME=False, batch_size=256, n_epochs=40)
		'''
	print "Running experiment 2..."
	train(train_matrix, "./models/basic_lstm_hsize300 lr01", "Basic LSTM hsize300 lr01",
		hidden_size=300, lr=0.001, RESUME=False, batch_size=256, n_epochs=40)
	print "Running experiment 3..."
	train(train_matrix, "./models/basic_lstm_hsize512 lr01", "Basic LSTM hsize512 lr01",
		hidden_size=512, lr=0.001, RESUME=False, batch_size=256, n_epochs=40)
	print ">>>>Learning rate"
	print "Running experiment 1..."
	train(train_matrix, "./models/basic_lstm_hsize256 lr01", "Basic LSTM hsize256 lr01",
		hidden_size=256, lr=0.005, RESUME=False, batch_size=256, n_epochs=40)
	print "Running experiment 2..."
	train(train_matrix,"./models/basic_lstm_hsize300 lr01", "Basic LSTM hsize300 lr01",
		hidden_size=300, lr=0.005, RESUME=False, batch_size=256, n_epochs=40)
	print "Running experiment 3..."
	train(train_matrix, "./models/basic_lstm_hsize512 lr01", "Basic LSTM hsize512 lr01",
		hidden_size=512, lr=0.005, RESUME=False, batch_size=256, n_epochs=40)
		'''
