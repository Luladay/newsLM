import util



if __name__ == '__main__':
	vocab_dict = util.openPkl("vocab_dict.pkl")
	reverse_dict = {}

	# print vocab_dict[util.UNK]
	#we have mapping from word to index
	# we now want index to word
	for word, index in vocab_dict.iteritems():		
		if word != util.UNK:
			reverse_dict[index] = word
	reverse_dict[0] = util.UNK

	# print reverse_dict[3]
	# print vocab_dict[reverse_dict[3]]

	util.dumpVar("reverse_dict.pkl", reverse_dict)


			



	# embed_matrix_mod = util.openPkl("embed_matrix_mod.pkl")
	# print embed_matrix_mod[0]
	# print embed_matrix_mod[1]
	# print embed_matrix_mod[len(embed_matrix_mod)-1]


