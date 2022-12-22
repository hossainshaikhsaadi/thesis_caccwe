import argparse
import gensim

parser = argparse.ArgumentParser(description='BLI evaluation')
parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--dict', type=str, help='whether to center embeddings or not')
args = parser.parse_args()


src_embeddings = gensim.models.KeyedVectors.load_word2vec_format(args.src_emb)
tgt_embeddings = gensim.models.KeyedVectors.load_word2vec_format(args.tgt_emb)

dictionary = open(args.dict,'r')
src_words = []
tgt_words = []
for line in dictionary:
	line = line.strip().split(" ")
	src_words.append(line[0].strip())
	tgt_words.append(line[1].strip())

unique_src = list(set(src_words))

src_dict = {}
for i in range(len(unique_src)):
	word = unique_src[i]
	start_index = src_words.index(word)
	word_list = []
	word_list.append(tgt_words[start_index])
	while True:
		try:
			start_index = src_words.index(word, start_index+1)
			word_list.append(tgt_words[start_index])
		except Exception as ex:
			break
	src_dict[word] = word_list

OOV_words = 0
p_at_1 = 0
p_at_5 = 0
found = 0
for src_word in unique_src:
    tgt_words = src_dict[src_word]
    try:
        src_emb = src_embeddings[src_word]
        similar_words = tgt_embeddings.similar_by_vector(src_emb, topn = 10) 

        similar_word_list = []
        for data in similar_words:
            word, similarity = data
            similar_word_list.append(word)

        set_1 = similar_word_list[0]
        set_2 = similar_word_list[0:5]

        if set_1 in tgt_words:
            p_at_1 = p_at_1 + 1
        
        found_in_set2 = 0
        for word in tgt_words:
            if word in set_2:
                found_in_set2 = 1
        if found_in_set2 == 1:
            p_at_5 = p_at_5 + 1
        
        found = found + 1

    except Exception as ex:
        OOV_words = OOV_words + 1
        continue

total = found + OOV_words
coverage = float(found/total)
print(p_at_1)
print(p_at_5)

print("P@1 score (including OOV words) : "+str(float(p_at_1/total)))
print("P@1 score (excluding OOV words) : "+str(float(p_at_1/found)))
print("P@5 score (including OOV words) : "+str(float(p_at_5/total)))
print("P@5 score (excluding OOV words) : "+str(float(p_at_5/found)))
print("OOV words : "+str(OOV_words))
print("In vocabulary : "+str(found))
print("Coverage : "+str(coverage))