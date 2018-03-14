import util
import numpy as np 



def addShort(string):
	return string + "_short"

def build_matrices(embed_matrix_filename, vocab_dict_filename, article_len, short=True):
	print "Opening embed_matrix..."
	embed_matrix = util.openPkl(embed_matrix_filename)
	print "Done opening embed_matrix!"
	print "Opening vocab_dict..."
	vocab_dict = util.openPkl(vocab_dict_filename)
	print "Done vocab_dict!"
	train_data_filename = ""
	test_data_filename = ""
	dev_data_filename = ""
	train_matrix_pkl = "train_matrix_rnn"
	train_labels_pkl = "train_labels_rnn"
	test_matrix_pkl = "test_matrix_rnn"
	test_labels_pkl = "test_labels_rnn"
	dev_matrix_pkl = "dev_matrix_rnn"
	dev_labels_pkl = "dev_labels_rnn"
	if short:
		train_data_filename = "Data/data_train_short.pkl"
		test_data_filename = "Data/data_test_short.pkl"
		dev_data_filename = "Data/data_dev_short.pkl"
		train_matrix_pkl = addShort(train_matrix_pkl)
		train_labels_pkl = addShort(train_labels_pkl)
		test_matrix_pkl = addShort(test_matrix_pkl)
		test_labels_pkl = addShort(test_labels_pkl)
		dev_matrix_pkl = addShort(dev_matrix_pkl)
		dev_labels_pkl = addShort(dev_labels_pkl)
	else:
		train_data_filename = "Data/data_train_full.pkl"
		test_data_filename = "Data/data_test_full.pkl"
		dev_data_filename = "Data/data_dev_full.pkl"

	train_matrix, train_labels = build_matrices_helper(train_data_filename, vocab_dict, embed_matrix, article_len)
	test_matrix, test_labels = build_matrices_helper(test_data_filename, vocab_dict, embed_matrix, article_len)
	dev_matrix, dev_labels = build_matrices_helper(dev_data_filename, vocab_dict, embed_matrix, article_len)
	
	
	util.dumpVar(train_matrix_pkl + ".pkl", train_matrix)
	util.dumpVar(train_labels_pkl + ".pkl", train_labels)

	util.dumpVar(test_matrix_pkl + ".pkl", test_matrix)	
	util.dumpVar(test_labels_pkl + ".pkl", test_labels)

	util.dumpVar(dev_matrix_pkl + ".pkl", dev_matrix)
	util.dumpVar(dev_labels_pkl + ".pkl", dev_labels)




def build_matrices_helper(data_filename, vocab_dict, embed_matrix, article_len):
	print "Opening " + data_filename + "!"
	data = util.openPkl(data_filename)
	print "Done "+ data_filename + "..."
	news_names = [util.nypost, util.bart, util.cnn, util.wpost, util.npr]
	n_articles = sum([len(data[news]) for news in news_names])
	print "total # of articles: " + str(n_articles)
	data_matrix = np.zeros((n_articles, article_len + len(news_names))) # do I need this?? + len(newsnames)
	row_num = 0	

	for i, news_source in enumerate(news_names):
		print "Adding articles from " + news_source
		all_articles = data[news_source]
		num_articles_so_far = 0
		for j, article in enumerate(all_articles):
			if num_articles_so_far % 1000 == 0:
				print "=======>" + "Adding article " + str(num_articles_so_far)
			indices = []
			if len(article) > 0:
				for token in article:
					if token.lower() in vocab_dict:
						indices.append(vocab_dict[token.lower()])
					else: # OOV word, map to UNK
						indices.append(vocab_dict[util.UNK])
				# print len(indices), indices

				article_vec = np.asarray(indices + (article_len - len(indices)) * [0])

				labels = np.zeros(len(news_names))
				labels[i] = 1
				article_vec = np.insert(article_vec, article_len, labels) #add label to end of matrix
				data_matrix[row_num] = article_vec 
				row_num += 1
			num_articles_so_far += 1
			
		

	# shuffle matrix bc everything was added in specific order	
	np.random.shuffle(data_matrix)
	# split matrix into examples (x) and labels (y)
	labels = data_matrix[:,article_len:]
	data_matrix = data_matrix[:,:article_len]
	return data_matrix, labels		



if __name__ == '__main__':

	#if running on full data (not short), change article_len param to be length of longest article
	build_matrices("embeddings_matrix.pkl", "vocab_dict.pkl", util.short_article_len)
	# dev_matrix = util.openPkl("dev_matrix_rnn_short.pkl")
	# dev_labels = util.openPkl("dev_labels_rnn_short.pkl")
	# print dev_matrix[0]
	# print dev_labels[0]	



	# test_matrix = util.openPkl("test_matrix_short.pkl")
	# test_labels = util.openPkl("test_labels_short.pkl")
	# print test_matrix.shape
	# print test_labels.shape