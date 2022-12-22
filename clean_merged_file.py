import argparse 

parser = argparse.ArgumentParser(description='clean merged corpus and alignment file')
parser.add_argument('--merged_file', type = str, help = 'path to merged corpus')
parser.add_argument('--alignment_file', type = str, help = 'path to alignment file')
args = parser.parse_args()

clean_merged_file = args.merged_file+".clean"
clean_alignment_file = args.alignment_file+".clean"

mrg_file = open(args.merged_file, 'r')
alm_file = open(args.alignment_file, 'r')
mrgc_file = open(clean_merged_file, 'w')
almc_file = open(clean_alignment_file, 'w')


for line, alignment in zip(mrg_file, alm_file):
    line = line.strip()
    lines = line.split("|||")
    src_words = lines[0].strip().split()
    tgt_words = lines[1].strip().split()

    if len(src_words) > 510 or len(tgt_words) > 510 :
        continue
    if len(src_words) == 0 or len(tgt_words) == 0 : 
        continue
    if len(alignment.strip().split()) == 0:
        continue

    merged_file_sentence = line + "\n"
    alignment_sentence = alignment.strip() + "\n"

    mrgc_file.write(merged_file_sentence)
    almc_file.write(alignment_sentence)



