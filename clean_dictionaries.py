import argparse

parser = argparse.ArgumentParser(description='Arguments for cleaning dictionaries')
parser.add_argument('--input_dict', type = str, help = 'path to input dictionary')
parser.add_argument('--output_dict', type = str, help = 'path to output dictionary')
args = parser.parse_args()

src_words = []
tgt_words = []
input_file = open(args.input_dict, 'r')
for line in input_file:
    line = line.strip()
    try:
        words = line.split(' ')
        if words[0].strip() != '' and words[1].strip() != '' and words[0].strip() != ' ' and words[1].strip() != ' ':
            src_words.append(words[0])
            tgt_words.append(words[1])
    except Exception as ex:
        words = line.split('\t')
        if words[0].strip() != '' and words[1].strip() != '' and words[0].strip() != ' ' and words[1].strip() != ' ':
            src_words.append(words[0])
            tgt_words.append(words[1])

output_file = open(args.output_dict, 'w')
for i in range(len(src_words)):
    sentence = src_words[i]+" "+tgt_words[i]+'\n'
    output_file.write(sentence)

