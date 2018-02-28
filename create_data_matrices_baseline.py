import util
import numpy as np 



def addShort(string):
	return string + "_short"

def build_matrices(embed_matrix_filename, vocab_dict_filename, short=True):
	print "Opening embed_matrix..."
	embed_matrix = util.openPkl(embed_matrix_filename)
	print "Done opening embed_matrix!"
	print "Opening vocab_dict..."
	vocab_dict = util.openPkl(vocab_dict_filename)
	print "Done vocab_dict!"
	train_data_filename = ""
	test_data_filename = ""
	dev_data_filename = ""
	train_matrix_pkl = "train_matrix"
	train_labels_pkl = "train_labels"
	test_matrix_pkl = "test_matrix"
	test_labels_pkl = "test_labels"
	dev_matrix_pkl = "dev_matrix"
	dev_labels_pkl = "dev_labels"
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

	train_matrix, train_labels = build_matrices_helper(train_data_filename, vocab_dict, embed_matrix)
	test_matrix, test_labels = build_matrices_helper(test_data_filename, vocab_dict, embed_matrix)
	dev_matrix, dev_labels = build_matrices_helper(dev_data_filename, vocab_dict, embed_matrix)
	
	
	util.dumpVar(train_matrix_pkl + ".pkl", train_matrix)
	util.dumpVar(train_labels_pkl + ".pkl", train_labels)

	util.dumpVar(test_matrix_pkl + ".pkl", test_matrix)	
	util.dumpVar(test_labels_pkl + ".pkl", test_labels)

	util.dumpVar(dev_matrix_pkl + ".pkl", dev_matrix)
	util.dumpVar(dev_labels_pkl + ".pkl", dev_labels)




def build_matrices_helper(data_filename, vocab_dict, embed_matrix):
	print "Opening " + data_filename + "!"
	data = util.openPkl(data_filename)
	print "Done "+ data_filename + "..."
	news_names = [util.nypost, util.bart, util.cnn, util.wpost, util.npr]
	n_articles = sum([len(data[news]) for news in news_names])
	print "total # of articles: " + str(n_articles)
	data_matrix = np.zeros((n_articles, util.glove_dimensions + 1))
	row_num = 0	

	for i, news_source in enumerate(news_names):
		print "Adding articles from " + news_source
		all_articles = data[news_source]
		num_articles_so_far = 0
		for article in all_articles:			
			if num_articles_so_far % 1000 == 0:
				print "=======>" + "Adding article " + str(num_articles_so_far)
			indices = []
			for token in article:
				if token.lower() in vocab_dict:
					indices.append(vocab_dict[token.lower()])
				else: # OOV word, map to UNK
					indices.append(vocab_dict[util.UNK])
			article_avg_vec = np.mean(embed_matrix[indices], 0)
			article_avg_vec = np.insert(article_avg_vec, util.glove_dimensions, i) #add label to end of matrix
			data_matrix[row_num] = article_avg_vec 
			row_num += 1
			num_articles_so_far += 1
			
		

	# shuffle matrix bc everything was added in specific order	
	np.random.shuffle(data_matrix)
	# split matrix into examples (x) and labels (y, last column)
	labels = data_matrix[:,util.glove_dimensions]
	data_matrix = data_matrix[:,:util.glove_dimensions]
	return data_matrix, labels		



if __name__ == '__main__':
	build_matrices("embeddings_matrix.pkl", "vocab_dict.pkl")
	# dev_matrix = util.openPkl("dev_matrix_short.pkl")
	# dev_labels = util.openPkl("dev_labels_short.pkl")
	# print dev_matrix.shape
	# print dev_labels.shape

	# test_matrix = util.openPkl("test_matrix_short.pkl")
	# test_labels = util.openPkl("test_labels_short.pkl")
	# print test_matrix.shape
	# print test_labels.shape