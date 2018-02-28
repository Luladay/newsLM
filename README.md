# newsLM

News sources: NY Post(0), Breitbart (1), CNN (2), Washington Post (3), NPR (4)


Data_file.pkl

This file is just a large dictionary with all of our data. All of the words of each article is included.

Data_test_full.pkl
Data_train_full.pkl
Data_test_full.pkl


These files have the first 7000 articles of each News Source in the test, the next 2000 in the train, and the next 2000 in the test. Each article had all of the words. An issue that we might get is that these articles were sorted by date...so we may want to mix and redo




Data_test_short.pkl

……

These files have only the first 100 words of each article, as by the TA’s suggestion


Data cleaning1.py

I used this to create the first dict with all of our data

Data cleaning2.py

This file splits the data into train test dev by a 60 20 20 split. Also creates the full and short files


--
create_vocab_embed_matrix.py 
Creates vocabulary dictionary from train file, maps word to integer representing row in embeddings matrix
Creates embeddings matrix from 50-dimensional GloVe vectors (dimensions for short vector: 62801 words (rows) x 50 (columns)

glove.py
File taken from cs224n assignment 1, very slightly modified
