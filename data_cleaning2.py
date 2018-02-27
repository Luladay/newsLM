#LET'S SPLIT THE DATA INTO TEST TRAIN DEV
import csv
import io
import nltk
import sys
import pickle
import math
csv.field_size_limit(sys.maxsize)




def truncate_list(dict_data, key, size):
	new_list = []
	for lists in dict_data[key]:
			lists = lists[:size] #take the first 100 words of each article
			new_list.append(lists)
	dict_data[key] = new_list
	return


with open('data_file.pkl', 'rb') as f:
    dict_data = pickle.load(f)


#total = len(dict_data['Washington_Post_articles']) the total length is 11114, this is the shortest onw

#hence we will take the 11000 frist articles


dict_data_train = {}
dict_data_dev   = {}
dict_data_test   = {}

for key in dict_data:
	dict_data_train[key] = dict_data[key][0    : 6660] 
	dict_data_dev[key]   = dict_data[key][6660 : 8880] 
	dict_data_test[key] = dict_data[key][8880 : 11100]





with open("data_train_full.pkl",'wb') as f:
    pickle.dump(dict_data_train,f)

with open("data_dev_full.pkl",'wb') as f:
    pickle.dump(dict_data_dev,f)


with open("data_test_full.pkl",'wb') as f:
    pickle.dump(dict_data_test,f)

#NOW LET'S WRITE THE FILES WITH JUST 100 WORDS IN EARCH ARTICLE

for key in dict_data:
	if(key == 'New_York_Post_articles'):
		truncate_list(dict_data_train, key, 100)
		truncate_list(dict_data_dev, key, 100)
		truncate_list(dict_data_test, key, 100)

	if(key == 'Breitbart_articles'):
		truncate_list(dict_data_train, key, 100)
		truncate_list(dict_data_dev, key, 100)
		truncate_list(dict_data_test, key, 100)

	if(key == 'CNN_articles'):
		truncate_list(dict_data_train, key, 100)
		truncate_list(dict_data_dev, key, 100)
		truncate_list(dict_data_test, key, 100)

	if(key == 'Washington_Post_articles'):
		truncate_list(dict_data_train, key, 100)
		truncate_list(dict_data_dev, key, 100)
		truncate_list(dict_data_test, key, 100)	

	if(key == 'NPR_articles'):
		truncate_list(dict_data_train, key, 100)
		truncate_list(dict_data_dev, key, 100)
		truncate_list(dict_data_test, key, 100)


with open("data_train_short.pkl",'wb') as f:
    pickle.dump(dict_data_train,f)

with open("data_dev_short.pkl",'wb') as f:
    pickle.dump(dict_data_dev,f)

with open("data_test_short.pkl",'wb') as f:
    pickle.dump(dict_data_test,f)



 


