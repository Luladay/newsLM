import pickle
import tensorflow as tf
import numpy as np
import util
from decimal import Decimal
from datetime import datetime
import matplotlib.pyplot as py
from tensorflow.python.layers import core as layers_core

today = datetime.now().strftime("%m-%d, %H:%M")

def generatePlots(x, y, xlabel, ylabel, title):	
	py.figure(figsize=(10,8))

	py.plot(x, y, color="blue")
	py.title(title, fontsize="large")
	py.xlabel(xlabel, fontsize="large")
	py.ylabel(ylabel, fontsize="large")
	py.savefig("graphs/" + title + " " + today + ".png", bbox_inches="tight")

   
def build_model(data_matrix, max_sequence_length=70, hidden_size=100, lr=0.005):
	n_features = util.glove_dimensions
	n_classes = 5
	max_gradient_norm = 5.
	max_sequence_length = max_sequence_length+1
	
	# add embedding layer!
	print "Opening embedding matrix..."	
	embed_matrix = util.openPkl("embed_matrix_mod.pkl")
	print "Done opening embedding matrix!"
	# x = tf.nn.embedding_lookup(embed_matrix, input_placeholder)

	# input_placeholder = tf.placeholder(tf.int32, shape=((None,None)))
	encoder_inputs_placeholder = tf.placeholder(tf.int32, shape=(max_sequence_length,None))
	decoder_inputs_placeholder = tf.placeholder(tf.int32, shape=(max_sequence_length,None))
	decoder_outputs_placeholder = tf.placeholder(tf.int32, shape=(max_sequence_length,None))	

	#assume encoder_inputs is size (max_time/len(sent, batch_size)
	batch_size = tf.shape(encoder_inputs_placeholder)[1]
	encoder_emb_inp = tf.nn.embedding_lookup(embed_matrix, encoder_inputs_placeholder)
	decoder_emb_inp = tf.nn.embedding_lookup(embed_matrix, decoder_inputs_placeholder)
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
	source_lengths = tf.ones([batch_size])*max_sequence_length
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_lengths, time_major=True, dtype=tf.float64)
	# print "encoder_state[0] ", encoder_state[0].shape

	
	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
	decoder_lengths = tf.ones([batch_size], dtype=tf.int32)*max_sequence_length
	# print "decoder_lengths type ", decoder_lengths.dtype
	helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)
	projection_layer = layers_core.Dense(util.vocab_size+1, use_bias=False)
	decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
	outputs, _ , _ = tf.contrib.seq2seq.dynamic_decode(decoder)
	logits = outputs.rnn_output
	
	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs_placeholder, logits=logits)
	target_weights = tf.ones((tf.shape(decoder_outputs_placeholder)[0], batch_size), dtype=tf.float64)
	train_loss = tf.reduce_mean(crossent*target_weights)
	params = tf.trainable_variables()
	gradients = tf.gradients(train_loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

	return logits, encoder_inputs_placeholder, decoder_inputs_placeholder, decoder_outputs_placeholder, train_op, train_loss


def build_classifier(data_matrix, data_labels, max_sequence_length=70, hidden_size=100, lr=0.005):
	n_features = util.glove_dimensions
	n_classes = 5
	max_grad_norm = 5.

	input_placeholder = tf.placeholder(tf.int32, shape=(max_sequence_length+1, None))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes))	

	print "Opening embedding matrix..."	
	embed_matrix = util.openPkl("embeddings_matrix.pkl")
	print "Done opening embedding matrix!"
	x = tf.nn.embedding_lookup(embed_matrix, input_placeholder)
	with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
		U = tf.get_variable("U", shape=[hidden_size, n_classes], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable("b", shape=[1, n_classes], dtype=tf.float64, initializer=tf.constant_initializer(0.0))

	with tf.variable_scope("", reuse=tf.AUTO_REUSE):
		rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
		rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=0.8)
		outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float64, time_major=True)

	h = final_state[1] #(max_sequence_length, hidden_size)
	pred = tf.matmul(h, U) + b

	loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred)
	loss_op = tf.reduce_mean(loss_op, 0)

	with tf.variable_scope("classify2", reuse=tf.AUTO_REUSE):
		params = tf.trainable_variables()
		gradients = tf.gradients(loss_op, params)
		clippied_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)	
		opt = tf.train.AdamOptimizer(learning_rate=lr)
		train_op = opt.apply_gradients(zip(clippied_gradients, params))
	return pred, input_placeholder, labels_placeholder, train_op, loss_op

def train_seq(data_matrix, save_path, title, max_sequence_length=70, hidden_size=100, lr=0.005, saved_model_path=None, RESUME=False, batch_size=100, n_epochs=30):
	tf.reset_default_graph()
	pred, encoder_inputs_placeholder, decoder_inputs_placeholder, decoder_outputs_placeholder, train_op, loss_op = build_model(data_matrix, 
		max_sequence_length=max_sequence_length, hidden_size=hidden_size, lr=lr)
	saver = tf.train.Saver()	
	avg_loss_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if RESUME:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, saved_model_path)
			print("Model restored.")

		minibatches = util.get_minibatches_seq(data_matrix, batch_size, max_sequence_length)
		for i in range(n_epochs):
			batch_loss_list = []
			print "Epoch " + str(i+1) + ": "
			for tup in minibatches:		
				_, loss = sess.run([train_op, loss_op], feed_dict={encoder_inputs_placeholder: np.transpose(tup[0]), 
				decoder_inputs_placeholder: np.transpose(tup[1]), decoder_outputs_placeholder: np.transpose(tup[0])})
				batch_loss_list.append(loss)
				print "loss: ", loss
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


def train_classifier(data_matrix, data_labels, save_path, seq2seq_model_path, title, max_sequence_length=70, hidden_size=100, lr=0.005, batch_size=60, n_epochs=30):
	tf.reset_default_graph()
	build_model(data_matrix)
	saver = tf.train.Saver()
	pred, input_placeholder, labels_placeholder, train_op, loss_op = build_classifier(data_matrix, data_labels, max_sequence_length)
	saver2 =tf.train.Saver()

	avg_loss_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, seq2seq_model_path)
		print("seq2seq model restored.")

		minibatches = util.get_minibatches_seq_test(data_matrix, data_labels, batch_size, max_sequence_length)
		for i in range(n_epochs):
			batch_loss_list = []
			print "Epoch " + str(i+1) + ": "
			for tup in minibatches:
				_, loss = sess.run([train_op, loss_op], feed_dict={input_placeholder: np.transpose(tup[0]), labels_placeholder: tup[1]})
				batch_loss_list.append(loss)
				print "loss: ", loss
			avg_loss_list.append(np.mean(batch_loss_list))
			print "=====>loss: " + str(avg_loss_list[i]) + " "
			if (i > 0) and (avg_loss_list[i] < avg_loss_list[i-1]):
				tmp_path = save_path + "--smallest loss"
				saver2.save(sess, tmp_path)
				print "New min loss at epoch %s! Model saved in path %s" % (str(i+1), tmp_path)
		saver2.save(sess, save_path)
  		print("Final model saved in path: %s" % save_path)

  		print tf.trainable_variables()

  	util.dumpVar("losses/ " + title + " " + today + ".pkl" , avg_loss_list)
  	generatePlots(range(len(avg_loss_list)), avg_loss_list, "Number of Epochs", "Cross-Entropy Loss", title)

def test_classifier(data_matrix, data_labels, saved_model_path, title, max_sequence_length=70, batch_size=60):
	tf.reset_default_graph() 
	pred, input_placeholder, labels_placeholder, train_op, loss_op = build_classifier(data_matrix, data_labels, max_sequence_length)
	saver = tf.train.Saver()

	loss_list = []
	label_list= []
	pred_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, saved_model_path)
		print("Model restored.")
		print tf.trainable_variables()			

		minibatches = util.get_minibatches_seq_test(data_matrix, data_labels, batch_size, max_sequence_length)
		for tup in minibatches:
			pred_temp, loss, labels_temp = sess.run([pred, loss_op, labels_placeholder], feed_dict={input_placeholder: np.transpose(tup[0]), labels_placeholder: tup[1]})
			for i, row in enumerate(pred_temp):
				pred_list.append(np.where(row == max(row))[0][0])
			for i, row in enumerate(labels_temp):
				label_list.append(np.where(row == max(row))[0][0])
			loss_list.append(loss)
		print "Loss: " + str(np.mean(loss_list)) + "\n"	

	util.outputConfusionMatrix(pred_list, label_list, "confusion_matrix " + title + " " + today)
	util.get_accuracy(pred_list, label_list)

def test_seq(data_matrix, saved_model_path, title, max_sequence_length=70, data_labels=None, batch_size=60):
	tf.reset_default_graph()
	pred, encoder_inputs_placeholder, decoder_inputs_placeholder, decoder_outputs_placeholder, _, loss_op = build_model(data_matrix)

	saver = tf.train.Saver()

	loss_list = []
	label_list= []
	pred_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, saved_model_path)
		print("Model restored.")

		minibatches = util.get_minibatches_seq_test(data_matrix, data_labels, batch_size, max_sequence_length)
		for tup in minibatches:
			pred_temp, loss, labels_temp = sess.run([pred, loss_op, labels_placeholder], feed_dict={input_placeholder: np.transpose(tup[0]), 
				labels_placeholder: tup[1]})
			for i, row in enumerate(pred_temp):
				pred_list.append(np.where(row == max(row))[0][0])
			for i, row in enumerate(labels_temp):
				label_list.append(np.where(row == max(row))[0][0])
			loss_list.append(loss)
		print "Loss: " + str(np.mean(loss_list)) + "\n"			

	# util.outputConfusionMatrix(pred_list, label_list, "confusion_matrix " + title + " " + today)
	# util.get_accuracy(pred_list, label_list)

if __name__ == '__main__':

	print "Opening train data..."
	train_matrix = util.openPkl("train_matrix_rnn_short.pkl")
	train_matrix = train_matrix[:1500, :]
	print "Done opening train data!"

	# train_labels = util.openPkl("train_labels_rnn_short.pkl")
	# print "Running experiment 1..."
	# print train_matrix.shape
	train_seq(train_matrix, "./models/seq2seq", "seq2seq", max_sequence_length=70,
		hidden_size=100, lr=0.005, saved_model_path="./models/seq2seq", RESUME=True, batch_size=60, n_epochs=10)

	# print "Opening dev data..."
	# dev_matrix = util.openPkl("dev_matrix_rnn_short.pkl")	
	# dev_labels = util.openPkl("dev_labels_rnn_short.pkl")
	# print "Done opening dev data!"
	# train_classifier(dev_matrix, dev_labels, "./models/seqlstm", "./models/seq2seq", "seqlstm", batch_size=60, n_epochs=2)
	# test_classifier(dev_matrix, dev_labels, "./models/seqlstm--smallest loss", "test", batch_size=60)