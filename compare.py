import util
import numpy as np

h300 = util.openPkl("losses/ Basic LSTM hidden_size 300 03-17, 12:44.pkl")
h256 = util.openPkl("losses/ Basic LSTM hidden_size 256 03-17, 12:48.pkl")
h512 = util.openPkl("losses/ Basic LSTM hidden_size 512 03-17, 12:46.pkl")

print "h256: ", min(h256), "at epoch ", np.argmin(h256)
print "h300: ", min(h300), "at epoch ", np.argmin(h300)
print "h512: ", min(h512), "at epoch ", np.argmin(h512)