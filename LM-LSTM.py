import pickle
import tensorflow as tf
import numpy as np
import util
from decimal import Decimal
from datetime import datetime
import matplotlib.pyplot as py


vocab_dict = util.openPkl("vocab_dict.pkl")

def build_model(data_matrix, data_labels):
	#n_features = util.glove_dimensions
	vocab_size = util.vocab_size
	lr = 0.001
	hidden_size = 256

	# add placeholders
	input_placeholder = tf.placeholder(tf.int32, shape=(None, util.short_article_len))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None, util.vocab_size))

	# add embedding layer!
	print "Opening embedding matrix..."
	embed_matrix = util.openPkl("embeddings_matrix.pkl")
	print "Done opening embedding matrix!"
	x = tf.nn.embedding_lookup(embed_matrix, input_placeholder)

	# build model
	U = tf.get_variable("U", shape=[hidden_size, util.vocab_size], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
	b = tf.get_variable("b", shape=[1, util.vocab_size], dtype=tf.float64, initializer=tf.constant_initializer(0.0))

	rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
	outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float64)
	pred = tf.matmul(final_state, U) + b


	loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred)
	loss_op = tf.reduce_mean(loss_op, 0)

	train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_op)
	return pred, input_placeholder, labels_placeholder, train_op, loss_op


def train(data_matrix, data_labels, save_path, title, RESUME=False, batch_size=256, n_epochs=30):
	if RESUME:
		tf.reset_default_graph()
	_, input_placeholder, labels_placeholder, train_op, loss_op = build_model(data_matrix, data_labels)
	saver = tf.train.Saver()
	avg_loss_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if RESUME:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, save_path)
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

  	generatePlots(range(len(avg_loss_list)), avg_loss_list, "Number of Epochs", "Cross-Entropy Loss", title)
  	util.dumpVar("losses/ " + title + " " + today + ".pkl" , avg_loss_list)


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

if __name__ == "__main__":
    #train_data_labels = util.openPkl("train_labels_rnn_short.pkl")
    vocab_dict = util.openPkl("vocab_dict.pkl")
    print(vocab_dict)
