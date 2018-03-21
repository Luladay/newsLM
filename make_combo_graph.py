import util
import basic_LSTM as lstm
import numpy as np


loss1 = util.openPkl("./losses/ seq2seq 03-20, 02:07.pkl")
loss2 = util.openPkl("./losses/ seq2seq 03-20, 06:52.pkl")
loss3 = util.openPkl("./losses/ seq2seq 03-20, 07:05.pkl")
loss4 = util.openPkl("./losses/ seq2seq 03-20, 23:38.pkl")
loss5 = util.openPkl("./losses/ seq2seq 03-21, 00:06.pkl")
loss6 = util.openPkl("./losses/ seq2seq 03-21, 00:45.pkl")
total_loss  = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
print len(total_loss)
print total_loss[0]
print total_loss[len(total_loss)-1]
perplexity = np.exp(total_loss)
print perplexity[0]
print perplexity[len(perplexity)-1]
# lstm.generatePlots(range(len(total_loss)), perplexity, "Number of Epochs", "Perplexity", "Seq2seq LSTM")

# l = util.openPkl("losses/ " + "Basic LSTM hsize256 lr01 03-18, 15:51.pkl")
# assert (l == total_loss)


# loss40 = util.openPkl("./losses/ Basic LSTM hidden_size 256 lr01 03-18, 03:03.pkl")
# loss10 = util.openPkl("./losses/ Basic LSTM hidden_size 256 lr01 +10 03-18, 03:36.pkl")
# total_loss  = loss40 + loss10
# lstm.generatePlots(range(len(total_loss)), total_loss, "Number of Epochs", "Cross Entropy Loss", "Basic LSTM (hidden_size=256, lr=0.001)")