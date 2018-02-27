import csv
import io
import nltk
import sys
import pickle

nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
csv.field_size_limit(sys.maxsize)


#READ THE DATA FROM ALL 3 FILES

#FILE 1


def populate_dict(dict_data, article):

	for row in article:
			if (row[3] == 'New York Times'):

			#ADD THE ARTICLE ID
				dict_data['New_York_Post_id'].append(row[1])


			#ADD THE TITLE
				dict_data['New_York_Post_title'].append(row[2])

			#ADD THE AUTHOR
				dict_data['New_York_Post_author'].append(row[4])

			#ADD THE DATE
				dict_data['New_York_Post_date'].append(row[5])

			#ADD THE TOKENIZED ARTICLE TO DIC
				word_list =  word_tokenize(row[9].decode('utf-8'))
				dict_data['New_York_Post_articles'].append(word_list)


			elif (row[3] == 'Breitbart'):

			#ADD THE ARTICLE ID
				dict_data['Breitbart_id'].append(row[1])


			#ADD THE TITLE
				dict_data['Breitbart_title'].append(row[2])

			#ADD THE AUTHOR
				dict_data['Breitbart_author'].append(row[4])

			#ADD THE DATE
				dict_data['Breitbart_date'].append(row[5])

			#ADD THE TOKENIZED ARTICLE TO DIC
				word_list =  word_tokenize(row[9].decode('utf-8'))
				dict_data['Breitbart_articles'].append(word_list)


			elif (row[3] == 'CNN'):

			#ADD THE ARTICLE ID
				dict_data['CNN_id'].append(row[1])


			#ADD THE TITLE
				dict_data['CNN_title'].append(row[2])

			#ADD THE AUTHOR
				dict_data['CNN_author'].append(row[4])

			#ADD THE DATE
				dict_data['CNN_date'].append(row[5])

			#ADD THE TOKENIZED ARTICLE TO DIC
				word_list =  word_tokenize(row[9].decode('utf-8'))
				dict_data['CNN_articles'].append(word_list)




			elif (row[3] == 'Washington Post'):


			#ADD THE ARTICLE ID
				dict_data['Washington_Post_id'].append(row[1])


			#ADD THE TITLE
				dict_data['Washington_Post_title'].append(row[2])

			#ADD THE AUTHOR
				dict_data['Washington_Post_author'].append(row[4])

			#ADD THE DATE
				dict_data['Washington_Post_date'].append(row[5])

			#ADD THE TOKENIZED ARTICLE TO DIC
				word_list =  word_tokenize(row[9].decode('utf-8'))
				dict_data['Washington_Post_articles'].append(word_list)

			elif (row[3] == 'NPR'):

			#ADD THE ARTICLE ID
				dict_data['NPR_id'].append(row[1])


			#ADD THE TITLE
				dict_data['NPR_title'].append(row[2])

			#ADD THE AUTHOR
				dict_data['NPR_author'].append(row[4])

			#ADD THE DATE
				dict_data['NPR_date'].append(row[5])

			#ADD THE TOKENIZED ARTICLE TO DIC
				word_list =  word_tokenize(row[9].decode('utf-8'))
				dict_data['NPR_articles'].append(word_list)





	return


dict_data = {'New_York_Post_articles': [], 'New_York_Post_id': [], 'New_York_Post_author' : [], 'New_York_Post_title': [], 'New_York_Post_date': [],

'Breitbart_articles': [], 'Breitbart_id': [], 'Breitbart_author': [], 'Breitbart_title': [], 'Breitbart_date': [],

'CNN_articles': [], 'CNN_id' : [], 'CNN_author': [], 'CNN_title': [], 'CNN_date' : [], 

'Washington_Post_articles': [], 'Washington_Post_id': [], 'Washington_Post_author': [], 'Washington_Post_title': [], 'Washington_Post_date': [],

'NPR_articles': [], 'NPR_id': [], 'NPR_author': [], 'NPR_title': [], 'NPR_date': []}





#FIRST FILE
with open('articles1.csv', 'rb') as csvfile1:
	try:
		read_article1 = csv.reader(csvfile1)

		populate_dict(dict_data, read_article1)
		

	finally:
		csvfile1.close()

#SECOND FILE
with open('articles2.csv', 'rb' ) as csvfile2:
	try:
		read_article2 = csv.reader(csvfile2)

		populate_dict(dict_data, read_article2)
		

	finally:
		csvfile2.close()

#THIRD FILE
with open('articles3.csv', 'rb') as csvfile3:
	try:
		read_article3 = csv.reader(csvfile3)

		populate_dict(dict_data, read_article3)
		

	finally:
		csvfile3.close()
#print(dict_data['CNN_author']) looks good! seems like some authors are empty spaces in the list though




with open("data_file.pkl",'wb') as f:
    pickle.dump(dict_data,f)




#PICKLE RICK!
#def dumpVar(filename, variable):
#	file = open(filename, 'w')
#	pickle.dump(variable, filename)
#	return
