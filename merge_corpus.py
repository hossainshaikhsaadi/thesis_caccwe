import argparse 

parser = argparse.ArgumentParser(description='merge two corpus for fast align')
parser.add_argument('--src_file', type = str, help = 'path to src language corpus')
parser.add_argument('--tgt_file', type = str, help = 'path to src language corpus')
parser.add_argument('--model', type = str, help = 'name of the language model')
parser.add_argument('--mrg_file', type = str, help = 'path to merged corpus')
args = parser.parse_args()

src_file = open(args.src_file, 'r')
tgt_file = open(args.tgt_file, 'r')
mrg_file = open(args.mrg_file, 'w')

for src_line, tgt_line in zip(src_file, tgt_file):
    
    src_line = src_line.strip()
    tgt_line = tgt_line.strip()

    if "\u200b" in src_line or "\u200b" in tgt_line:
        continue
    if "\u202a" in src_line or "\u202a" in tgt_line:
        continue
    if "\u202c" in src_line or "\u202c" in tgt_line:
        continue
    if "\u200d" in src_line or "\u200d" in tgt_line:
        continue
    if "\u200e" in src_line or "\u200e" in tgt_line:
	    continue
    if "\u200c" in src_line or "\u200c" in tgt_line:
        continue
    if "\ufeff" in src_line or "\ufeff" in tgt_line:
        continue
    if "�" in src_line or "�" in tgt_line:
        continue
    if "#" in src_line or "#" in tgt_line:
        continue
    if "##" in src_line or "##" in tgt_line:
        continue
    if "@@" in src_line or "@@" in tgt_line:
        continue
    if "@" in src_line or "@" in tgt_line:
        continue
    
    mrg_sentence = src_line + " ||| "+tgt_line+"\n"
    mrg_file.write(mrg_sentence)

src_file.close()
tgt_file.close()
mrg_file.close()