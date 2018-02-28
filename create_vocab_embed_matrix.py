import pickle
import time
import glove as glove
import numpy as np


nypost = "New_York_Post_articles"
bart = "Breitbart_articles"
cnn = "CNN_articles"
wpost = "Washington_Post_articles"
npr = "NPR_articles"

UNK = '<UNK>'
dimensions = 50

def dumpVar(filename, obj):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	
def openPkl(filename):
	var = open(filename, "rb")
	return pickle.load(var)


def create_vocab_dict_helper(train_data, news_names):
	vocab_dict = {}	
	vocab_dict[UNK] = 1
	num_tokens_so_far = 1
	for news_source in news_names:
		print "Adding articles from " + news_source
		all_articles = train_data[news_source]
		num_articles_so_far = 0
		for article in all_articles:			
			if num_articles_so_far % 100 == 0:
				print "=======>" + "Adding article " + str(num_articles_so_far)
			for token in article:
				token = token.lower()
				if token not in vocab_dict:
					vocab_dict[token] = num_tokens_so_far
					num_tokens_so_far += 1
					print num_tokens_so_far	
			num_articles_so_far += 1
	dumpVar("vocab_dict.pkl", vocab_dict)

def create_vocab_dict():
	print "Opening train_data..."
	train_short = openPkl("Data/data_train_short.pkl")
	print "Done opening train_data!"
	news_names = [nypost, bart, cnn, wpost, npr]
	print "Creating vocab_dict now..."
	create_vocab_dict_helper(train_short, news_names)
	print "Done creating vocab_dict!"


def create_embed_matrix():
	print "Opening vocab_dict..."	
	vocab_dict = openPkl("vocab_dict.pkl")
	print "Done opening vocab_dict!"
	print "Creating embed_matrix..."
	vocab_size = len(vocab_dict)
	embed_matrix = glove.loadWordVectors(vocab_dict, dimensions=dimensions)	
	print "Done creating embed_matrix!"
	print "Cleaning up the embeddings_matrix..."

	# words w/o glove vectors never got updated in matrix
	# should have empty field, so give them 0s and remap to UNK
	for word in vocab_dict:
		embed_matrix_word_index = vocab_dict[word]
		if len(embed_matrix[embed_matrix_word_index]) < dimensions:
			embed_matrix[embed_matrix_word_index] = [float(0) for x in range(dimensions)]
			vocab_dict[word] = vocab_dict[UNK]

	print "Done cleaning up the data!"
	dumpVar("embeddings_matrix.pkl", embed_matrix)


if __name__ == '__main__':

	# create_vocab_dict()

	# create_embed_matrix()
	# embed_matrix = openPkl("embeddings_matrix.pkl")
	# print embed_matrix.shape
