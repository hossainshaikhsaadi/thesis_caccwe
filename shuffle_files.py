import random
import argparse 
import numpy as np

random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='clean merged corpus and alignment file')
parser.add_argument('--merged_file', type = str, help = 'path to merged corpus')
parser.add_argument('--alignment_file', type = str, help = 'path to alignment file')
args = parser.parse_args()

mrg_file = open(args.merged_file, 'r')
alm_file = open(args.alignment_file, 'r')

sentences = []
alignments = []

for line1, line2 in zip(mrg_file, alm_file):
	line1 = line1.strip()
	line2 = line2.strip()

	sentences.append(line1)
	alignments.append(line2)

mrg_file.close()
alm_file.close()

mrg_file = open(args.merged_file + ".shuffled", 'w')
alm_file = open(args.alignment_file + ".shuffled", 'w')

shuffled_list = list(zip(sentences, alignments))
random.shuffle(shuffled_list)
shuffled_sentences, shuffled_alignments = zip(*shuffled_list)

for sentence1, sentence2 in zip(shuffled_sentences, shuffled_alignments):
	mrg_file.write(sentence1+"\n")
	alm_file.write(sentence2+"\n")

mrg_file.close()
alm_file.close()