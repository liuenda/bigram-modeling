# import modules & set up logging
import gensim, logging
from gensim.models import *

fname='./model-en/W2Vmodle.bin'
model = Word2Vec.load(fname)

