import numpy as np


# Taken from cs224n assignment 1
# This is not my original code, some modifications were made

# Given vocab dictionary, creates embedding matrix 
# Remaps

DEFAULT_FILE_PATH = "Data/glove.6B.50d.txt"

def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors"""
    lines_so_far = 0
    wordVectors = np.zeros((len(tokens), dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            if lines_so_far % 1000 == 0:
                print "======> passed line " + str(lines_so_far)
            lines_so_far += 1
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = np.asarray(data)
    return wordVectors