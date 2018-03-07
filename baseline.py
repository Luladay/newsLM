import pickle
import tensorflow as tf
import numpy as np

#just trying this out
#just making changes
# change the filenames if the pickle files are in a Data folder
train_matrix = util.openPkl("train_matrix_short.pkl")
train_labels = util.openPkl("train_labels_short.pkl")
dev_matrix = util.openPkl("dev_matrix_short.pkl")
dev_labels = util.openPkl("dev_labels_short.pkl")
test_matrix = util.openPkl("test_matrix_short.pkl")
test_labels = util.openPkl("test_labels_short.pkl")




n_samples = train_matrix.shape[0]
n_features = train_matrix.shape[1] #should be util.glove_dimensions
n_classes = 5
batch_size = 60
n_epochs = 50
lr = 0.001


input_placeholder = tf.placeholder(tf.float32, shape=(n_samples, n_features))
labels_placeholder = tf.placeholder(tf.int32, shape=(1, n_classes))

W = tf.Variable(tf.zeros([n_features, n_classes], dtype=tf.float32))
b = tf.Variable(tf.zeros([n_classes, 1], dtype=tf.float32))

xW = tf.matmul(input_placeholder, W)
pred = tf.transpose(tf.transpose(xW) + b)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred)
loss = tf.reduce_mean(loss, 0) #not sure if we only needed this before in the hw b/c we were training in batches??



train_op = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(loss)



with tf.Graph().as_default() as graph:
	# Build the model and add the variable initializer op
	init_op = tf.global_variables_initializer()
	# Finalizing the graph causes tensorflow to raise an exception if you try to modify the graph
	# further. This is good practice because it makes explicit the distinction between building and
	# running the graph.
	graph.finalize()

	# Create a session for running ops in the graph
	with tf.Session(graph=graph) as sess:
		# Run the op to initialize the variables.
		sess.run(init_op)
		# Fit the model
		losses = model.fit(sess, inputs, labels)
