import pickle


UNK = '<UNK>'

nypost = "New_York_Post_articles"
bart = "Breitbart_articles"
cnn = "CNN_articles"
wpost = "Washington_Post_articles"
npr = "NPR_articles"

nypost_label = 0
bart_label = 1
cnn_label = 2
wpost_label = 3
npr_label = 4

short_article_len = 100


glove_dimensions = 50


def dumpVar(filename, obj):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def openPkl(filename):
	var = open(filename, "rb")
	return pickle.load(var)

