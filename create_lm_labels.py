import util
import numpy as np

'''
this file aims to create a matrix to store the labels for the language model.
Each row of this matrix will contain a 1 at the index of the number stored in
the vocab_dict map. Each row represents the true label of each successive word

thus for our short news articles that have 100 words, the labels matrix will
have 100 rows and size(vocab_dict) columns

this is needed to calculate the loss in the language model LSTM

vocab_dict = util.openPkl("vocab_dict.pkl")

def create_lm_data_labels(data,labels,filename):
    print "Opening data..."
    data_matrix = util.openPkl('train_matrix_rnn_short.pkl')
    print "Done opening data!"

    lm_matrix = data_matrix[0:1000, :]

    lm_labels =





    article_names = [util.nypost, util.bart, util.cnn, util.wpost, util.npr ]
    list_indexes = []
    for names in article_names:
        for articles in dict_data[names]:
            if len(articles) > 0 :
                for word in articles:
                    if word not in vocab_dict:
                        list_indexes.append(0)
                    else:
                        list_indexes.append(vocab_dict[word])
    # the following lines place a 1 at row  i of the matrix
    #wherever list_index(i) is.
    '''
    '''
    matrix = np.zeros((len(list_indexes), util.vocab_size))
    matrix[np.arange(len(list_indexes)), list_indexes] = 1


    util.dumpVar(filename, list_indexes)

            #let's create the labels matrix for the train data



if __name__ == "__main__":
    print "creating train lm labels..."
    create_lm_labels('data_train_short.pkl', 'train_lm_labels.pkl')
    print "done creating train lm labels!"

    print "creating dev lm labels..."
    create_lm_labels('data_dev_short.pkl', 'dev_lm_labels.pkl')
    print "done creating dev lm labels!"

    print "creating test lm labels..."
    create_lm_labels('data_test_short.pkl', 'test_lm_labels.pkl')
    print "done creating test lm labels!"
