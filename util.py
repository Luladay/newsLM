import pickle


UNK = '<UNK>'

nypost = "New_York_Post_articles"
bart = "Breitbart_articles"
cnn = "CNN_articles"
wpost = "Washington_Post_articles"
npr = "NPR_articles"

glove_dimensions = 50


def dumpVar(filename, obj):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def openPkl(filename):
	var = open(filename, "rb")
	return pickle.load(var)