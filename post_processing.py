import argparse 

from transformers import BertTokenizer
from transformers import DistilBertTokenizer
from transformers import AlbertTokenizer
from transformers import ElectraTokenizer
from transformers import XLMRobertaTokenizer

src_tokenizer = dict()
tgt_tokenizer = dict()
multilingual_tokenizer = dict()


src_tokenizer[0] = BertTokenizer.from_pretrained('sagorsarker/bangla-bert-base', do_lower_case = False)
src_tokenizer[1] = ElectraTokenizer.from_pretrained('/media/saadi/Saadi/HiWi/helper_folder/Master_Thesis/pretrained_models/bangla_electra', 
                                                do_lower_case = False)

tgt_tokenizer[0] = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = False)
tgt_tokenizer[1] = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', do_lower_case = False) 

multilingual_tokenizer[0] = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased', do_lower_case = True)
multilingual_tokenizer[1] = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case = False)
multilingual_tokenizer[2] = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case = False)
multilingual_tokenizer[3] = AlbertTokenizer.from_pretrained('ai4bharat/indic-bert', do_lower_case = False)

parser = argparse.ArgumentParser(description='merge two corpus for fast align')
parser.add_argument('--mrg_file', type = str, help = 'path to src language corpus')
parser.add_argument('--alm_file', type = str, help = 'path to src language corpus')
parser.add_argument('--nmrg_file', type = str, help = 'path to src language corpus')
parser.add_argument('--nalm_file', type = str, help = 'path to src language corpus')
parser.add_argument('--num_sent', type = int, help = 'path to src language corpus')
args = parser.parse_args()

mrg_file = open(args.mrg_file, 'r')
alm_file = open(args.alm_file, 'r')
nmrg_file = open(args.nmrg_file, 'w')
nalm_file = open(args.nalm_file, 'w')

sent_count = 0

for line_mrg, line_alm in zip(mrg_file, alm_file):

    lines = line_mrg.strip().split("|||")
    src_words = lines[0].strip().split()
    tgt_words = lines[1].strip().split()

    error_src = 0
    for word in src_words:
        for i in range(len(src_tokenizer)):   
            word_tokens = src_tokenizer[i].tokenize(word)
            if len(word_tokens) <= 0:
                error_src = 1
        if error_src == 1:
            break
        for i in range(len(multilingual_tokenizer)):   
            word_tokens = multilingual_tokenizer[i].tokenize(word)
            if len(word_tokens) <= 0:
                error_src = 1
        if error_src == 1:
            break
    if error_src == 1:
        continue

    error_tgt = 0
    for word in tgt_words:
        for i in range(len(tgt_tokenizer)):   
            word_tokens = tgt_tokenizer[i].tokenize(word)
            if len(word_tokens) <= 0:
                error_tgt = 1
        if error_tgt == 1:
            break
        for i in range(len(multilingual_tokenizer)):   
            word_tokens = multilingual_tokenizer[i].tokenize(word)
            if len(word_tokens) <= 0:
                error_tgt = 1
        if error_tgt == 1:
            break
    if error_tgt == 1:
        continue

    if len(line_alm.strip().split()) < 1:
        continue

    merged_sentence = line_mrg.strip() + "\n"
    alignment = line_alm.strip() + "\n"

    nmrg_file.write(merged_sentence)
    nalm_file.write(alignment)
    sent_count = sent_count + 1
    if sent_count > args.num_sent:
        break

mrg_file.close()
alm_file.close()
nmrg_file.close()
nalm_file.close()