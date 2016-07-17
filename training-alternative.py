# import modules & set up logging
import gensim, logging
#from gensim.models import Word2Vec
from gensim.models import *
import os
fname='W2Vmodle'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
	def __init__(self, dirname):
		#print "dirname=",dirname
		self.dirname = dirname

	def __iter__(self):
			for line in open(self.dirname):
				yield line.split()

sentences = MySentences('../TrainingData/SampleData.txt') # a memory-friendly iterator
model = Word2Vec(sentences,size=200)  # default value is 100

model.save(fname) # save the model
model.save_word2vec_format('vectors.txt', binary=False)
model = Word2Vec.load(fname) # you can continue training with the loaded model!
