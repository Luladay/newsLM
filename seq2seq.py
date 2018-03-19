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
	# py.plot(x, testAccReg, "b--", label='Test Accuracy (Lyrics)')
	# py.plot(x, trainAccAug, label='Train Accuracy (Lyrics + Audio)', color="red")
	# py.plot(x, testAccAug, linestyle="--", color="red", label='Test Accuracy (Lyrics + Audio)')
	# py.figlegend()
	py.title(title, fontsize="large")
	py.xlabel(xlabel, fontsize="large")
	py.ylabel(ylabel, fontsize="large")
	py.savefig("graphs/" + title + " " + today + ".png", bbox_inches="tight")

   
#since test() relies on default value of hidden_size and lr, be sure to update default value once it's tuned!!!!
def build_model(data_matrix, batch_size=256, hidden_size=256, lr=0.001):
	n_features = util.glove_dimensions
	n_classes = 5
	max_gradient_norm = 5.
	sentence_len = util.short_article_len+1
	
	# add embedding layer!
	print "Opening embedding matrix..."	
	embed_matrix = util.openPkl("embed_matrix_mod.pkl")
	print "Done opening embedding matrix!"
	# x = tf.nn.embedding_lookup(embed_matrix, input_placeholder)

	# input_placeholder = tf.placeholder(tf.int32, shape=((None,None)))
	encoder_inputs_placeholder = tf.placeholder(tf.int32, shape=(sentence_len,None))
	decoder_inputs_placeholder = tf.placeholder(tf.int32, shape=(sentence_len,None))
	decoder_outputs_placeholder = tf.placeholder(tf.int32, shape=(sentence_len,None))
	# labels_placeholder = tf.placeholder(tf.int32, shape=(sentence_len,None))
	encoder_inputs = encoder_inputs_placeholder
	print "encoder_inputs ", encoder_inputs.shape
	decoder_inputs = decoder_inputs_placeholder
	print "decoder_inputs ", decoder_inputs.shape
	decoder_outputs = decoder_outputs_placeholder
	print "decode outputs ", decoder_outputs.shape	

	#assume encoder_inputs is size (max_time/len(sent, batch_size)
	encoder_emb_inp = tf.nn.embedding_lookup(embed_matrix, encoder_inputs)
	#same assumptions for decoder
	decoder_emb_inp = tf.nn.embedding_lookup(embed_matrix, decoder_inputs)
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
	#encoder_outputs =(max_time, batch_size, hidden_size), encoder_state = (batch_size, hidden_size)
	source_lengths = tf.ones(batch_size)*sentence_len
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_lengths, time_major=True, dtype=tf.float64)
	print "encoder_outputs ", encoder_outputs.shape
	print "encoder_state[0] ", encoder_state[0].shape
	print "encoder_state[1] ", encoder_state[1].shape
	
	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
	#i think decoder_lengths is the length of each decoded sequence?
	decoder_lengths = tf.ones(batch_size)*sentence_len
	helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)
	projection_layer = layers_core.Dense(util.vocab_size, use_bias=False)
	decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
	outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
	assert(not True)
	logits = outputs.rnn_output
	
	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs)
	#what i call loss_op
	train_loss = (tf.reduce(sum(crossent*tf.ones((decoder_outputs.shape)))) / batch_size)
	params = tf.trainable_variables()
	gradients = tf.gradients(train_loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
	optimizer = tf.train.AdamOptimizer(learnign_rate=lr)
	train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

	return pred, encoder_inputs_placeholder, decoder_inputs_placeholder, decoder_outputs_placeholder, train_op, loss_op

	# output_projection = None
	# x = tf.get_variable("proj_w", [hidden_size, util.vocab_size])
	# w_t = tf.transpose(w)
	# b = tf.get_variable("proj_b", [util.vocab_size])
	# output_projection = (w,b)

	# encoder_inputs = []
	# decoder_inputs = []
	# target_weights = [] #not sure if we need this

	# #I think inputs are tuples (encoder_batch, decode_batch) ??
	# encoder_inputs.append(tf.placeholder(tf.int32, shape=[None]))
	# decoder_inputs.append(tf.placeholder(tf.in32, shape=[None]))
	# target_weights.append(tf.placeholder(itf.float32, shape=[None]))

	 
	# # add placeholders
	# input_placeholder = tf.placeholder(tf.int32, shape=(None, util.short_article_len))
	# labels_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes))


	# # build model
	# U = tf.get_variable("U", shape=[hidden_size, n_classes], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
	# b = tf.get_variable("b", shape=[1, n_classes], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
    
	# rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
	# rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=0.5)
	# outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float64)

	# h = final_state[1]
	# pred = tf.matmul(h, U) + b

	# loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred)
	# loss_op = tf.reduce_mean(loss_op, 0)

	# train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_op)
	# return pred, input_placeholder, labels_placeholder, train_op, loss_op


def train(data_matrix, save_path, title, hidden_size=256, lr=0.001, saved_model_path=None, RESUME=False, batch_size=256, n_epochs=30):
	tf.reset_default_graph()
	pred, encoder_inputs_placeholder, decoder_inputs_placeholder, decoder_outputs_placeholder, train_op, loss_op = build_model(data_matrix, batch_size=256, hidden_size=hidden_size, lr=lr)	
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
				_, loss = sess.run([train_op, loss_op], feed_dict={encoder_inputs_placeholder: tf.transpose(tup[0]), 
				decoder_inputs_placeholder: tf.transpose(tup[1]), decoder_outputs_placeholder: tf.transpose(tup[0])})
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

# currently not modified much!
def test(data_matrix, saved_model_path, title, batch_size=256):
	tf.reset_default_graph()
	pred, input_placeholder, labels_placeholder, _, loss_op = build_model(data_matrix)
	saver = tf.train.Saver()
	loss_list = []
	label_list= []
	pred_list = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, saved_model_path)
		print("Model restored.")

		minibatches = util.get_minibatches_lm(data_matrix, batch_size)
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
	# train_labels = util.openPkl("train_labels_rnn_short.pkl")
	print "Done opening train data!"
	# print "Running experiment 1..."
	train(train_matrix, "./models/seq2seq", "seq2seq", 
		hidden_size=256, lr=0.001, RESUME=False, batch_size=256, n_epochs=5)
	# print "Running experiment"
	# train(train_matrix, train_labels, "./models/basic_lstm_cell drop05", "Basic LSTM cell drop05", 
		# hidden_size=256, lr=0.001, RESUME=False, batch_size=256, n_epochs=20)