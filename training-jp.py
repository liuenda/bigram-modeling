import gensim, logging
from gensim.models import *
import os

output_filename='./model-jp/W2Vmodle.bin'
output_filename2='./model-jp/W2Vmodle.txt'
input_filename="../cleaned2_tag_jp.txt"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [line.split() for line in open(input_filename)]
model = Word2Vec(sentences,size=200,workers=8,min_count=1)  # default value is 100,1, parallel required Cython installed

model.save(output_filename) # save the model
model.save_word2vec_format(output_filename2, binary=False)
#model = Word2Vec.load(output_filename) # you can continue training with the loaded model!
