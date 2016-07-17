# coding:utf-8
from gensim.models import Phrases
import sys  
import time
from  nltk import word_tokenize
start_time = time.time()
reload(sys)  
sys.setdefaultencoding('utf8')

def load_paras_en(input_filename='../for_phraseDetect_en.txt'):
	paras=[]
	with open(input_filename) as data_file:
		print "start pharse detection:"
		for (index,line) in enumerate(data_file):
			tokens=word_tokenize(line)
			paras.append(tokens)
			# Print progress
			# if index in range(5000,60001,5000):
			# 	print "Now read the line No.:"+str(index)
			# 	print("--- %s seconds ---" % (time.time() - start_time))
		print "finish loading text resources in: "
		print("--- %s seconds ---" % (time.time() - start_time))
	return paras

# Train the pharse detector all in once
def train_detector(paras):	
	print "start to:Train the pharse detector all in once"
	bigram = Phrases(paras)
	print("--- %s seconds ---" % (time.time() - start_time))
	return bigram


# This is the function expected to be envoked
def phrase_detector_en(input_filename='../for_phraseDetect_en.txt'):
	paras=load_paras_en(input_filename)
	detector=train_detector(paras)
	return detector


def phrase_detector2_en(input_filename='../for_phraseDetect_en.txt')
	detector=phrase_detector_en(input_filename)
	detector=train_detector(paras)
	return detector2


if __name__=='__main__':
	# Load file
	paras=load_paras_en(input_filename)

	# Train bi-detector
	detector=train_detector(paras)

	# Define file name
	input_filename="../cleaned2_tag_en.txt"
	output_filename="cleaned2_tag_phrase_en.txt"

	sentences = [line.split() for line in open(input_filename)]
	sentences = detector[sentences]
	
	f=open(output_filename,'w')
	o=[f.write(' '.join(line)+'\n') for line in sentences]
	f.close()

	# Train tri-detector
	output_filename2="cleaned2_tag_phrase2_en.txt"
	detector2=train_detector(sentences)
	sentences2 = detector2[sentences]

	f=open(output_filename2,'w')
	o=[f.write(' '.join(line)+'\n') for line in sentences2]
	f.close()
# sent = [u'the', u'mayor', u'of', u'new', u'york', u'be', u'there']


# if __name__=='__main__':
# 	# Load file
# 	paras=load_paras_en(input_filename)

# 	# Train bi-detector
# 	detector=train_detector(paras)

# 	# Define file name
# 	input_filename="../cleaned2_tag_en.txt"
# 	output_filename="cleaned2_tag_phrase_en.txt"

# 	sentences = [line.split() for line in open(input_filename)]
# 	sentences = detector[sentences]
	
# 	f=open(output_filename,'w')
# 	o=[f.write(' '.join(line)+'\n') for line in sentences]
# 	f.close()

# 	# Train tri-detector
# 	output_filename2="cleaned2_tag_phrase2_en.txt"
# 	detector2=train_detector(sentences)
# 	sentences2 = detector2[sentences]

# 	f=open(output_filename2,'w')
# 	o=[f.write(' '.join(line)+'\n') for line in sentences2]
# 	f.close()
# # sent = [u'the', u'mayor', u'of', u'new', u'york', u'be', u'there']