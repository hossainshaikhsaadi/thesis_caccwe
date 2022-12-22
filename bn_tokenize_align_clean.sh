



#bangla to english preprocessings steps for cao style evaluation

python bengali_process.py

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en <  \
/home/hsaadi/helper_folder/data/language_corpus/original_corpus.en \
>  /home/hsaadi/helper_folder/data/language_corpus/original_corpus.en.tokennized

python merge_corpus.py --src_file  /home/hsaadi/helper_folder/data/language_corpus/original_corpus.bn.tokenized \
--tgt_file /home/hsaadi/helper_folder/data/language_corpus/original_corpus.en.tokenized \
--mrg_file ./data/language_corpus/merged.bn-en -

fast_align/build/fast_align -i  /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en -d -o -v \
> ./data/language_corpus/merged.bn-en.align

fast_align/build/fast_align -i  /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en -d -o -v -r \
>  ./data/language_corpus/merged.bn-en.reverse.align

fast_align/build/atools -i  /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.align -j \
 /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.reverse.align -c intersect \
 >  /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.alignment

python clean_merged_file.py --merged_file  /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en \
--alignment_file  /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.alignment

python shuffle_files.py --merged_file  /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.clean \
--alignment_file  /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.alignment.clean

python post_processing.py --mrg_file /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.clean.shuffled \
--alm_file /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.alignment.clean.shuffled \
--nmrg_file /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.clean.shuffled.albert \
--nalm_file /home/hsaadi/helper_folder/data/language_corpus/merged.bn-en.alignment.clean.shuffled.albert \
--num_sent 350000

