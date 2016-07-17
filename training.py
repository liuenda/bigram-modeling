#!/usr/bin/python
import gensim, logging
from gensim.models import *
import os
import sys
import phrase_det
import time
# output_filename='./Model/W2Vmodle.bin'
# output_filename2='./Model/W2Vmodle.txt'
# input_filename="../cleaned2_tag_en.txt"

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentences = [line.split() for line in open(input_filename)]
# model = Word2Vec(sentences,size=200,workers=8,min_count=1)  # default value is 100,1, parallel required Cython installed

# model.save(output_filename) # save the model
# model.save_word2vec_format(output_filename2, binary=False)
# #model = Word2Vec.load(output_filename) # you can continue training with the loaded model!


def train_model(lang,phrase_flag=1):
	# Define file name by language:
	phrase_keyword=""
	if phrase_flag==1:
		phrase_keyword="_phrase"

	input_filename="../cleaned2_tag_en.txt"
	input_filename2='../for_phraseDetect_en.txt'
	output_filename='./model-'+lang+phrase_keyword+'/W2Vmodle.bin'
	output_filename2='./model-'+lang+phrase_keyword+'/W2Vmodle.txt'	

	print "Prepare to train the model with: "+lang+phrase_keyword

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	sentences = [line.split() for line in open(input_filename)]
	
	if phrase_flag==1:
		detector=phrase_det.phrase_detector_en(input_filename2)
		sentences = detector[sentences]
		print "Sentences have been transformed into phrased version"

	model = Word2Vec(sentences,size=200,workers=8,min_count=1)  # default value is 100,1, parallel required Cython installed

	model.save(output_filename) # save the model
	model.save_word2vec_format(output_filename2, binary=False)
	#model = Word2Vec.load(output_filename) # you can continue training with the loaded model!
	return model



if __name__=="__main__":
	# n_argu=len(sys.argv)
	start_time = time.time()

	argu=' '.join(sys.argv)
	print "the argument is ",str(argu)

	if 'phrase' in argu:
		print "Use phrase detection for model training"
		phrase_flag=1
	if 'jp' in argu:
		print "Prepare to train Japanese model with phrase flag="+phrase_flag
		model_jp=train_model('jp',phrase_flag)
	if 'en' in argu:
		print "Prepare to train English model with phrase flag="+phrase_flag
		model_en=train_model('en',phrase_flag)
	if len(sys.argv)==1:
		print "Prepare to train English (default) model with phrase detection enable (phrase_flag=1)"
		model=train_model('en')
	print("--- %s seconds ---" % (time.time() - start_time))
