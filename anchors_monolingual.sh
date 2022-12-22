

model=bert
src_lang=bn
tgt_lang=en
prefix=bbert
batch_size=5
max_sent=100
max_len=256
device=cuda
src_lower_case=False
tgt_lower_case=False
src_model_path=sagorsarker/bangla-bert-base
tgt_model_path=bert-base-uncased
total_sentences=200000000

lm_path=None #mbert.bn-de-en.pt #change
base_directory=/home/hsaadi/helper_folder
single_layer_path=${base_directory}/saved_models/test.mbert.transformer.bn-de-en.pt
adapter_path=${base_directory}/saved_adapters/test_adapter/
adapter_name=adapter_for_test
#--lm_path /home/hsaadi/helper_folder/saved_models/${lm_path}
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name} \
#--single_layer ${single_layer_path}



lang_pair=${src_lang}-${tgt_lang}
src_wiki=${src_lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt
tgt_wiki=${tgt_lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt

python generate_avg_anchors.py --input_dict /home/hsaadi/helper_folder/data/words/${src_lang}.txt \
--lang ${src_lang} --input_file  /home/hsaadi/helper_folder/${src_wiki}  \
--output_file  /home/hsaadi/helper_folder/embeddings/${prefix}.${lang_pair}.${src_lang} \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${src_model_path} --temp_file temp/temp.txt \
--lower_case False --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device} 

python generate_avg_anchors.py --input_dict /home/hsaadi/helper_folder/data/words/${tgt_lang}.txt \
--lang ${tgt_lang} --input_file  /home/hsaadi/helper_folder/${tgt_wiki}  \
--output_file  /home/hsaadi/helper_folder/embeddings/${prefix}.${lang_pair}.${tgt_lang} \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${tgt_model_path} --temp_file temp/temp.txt \
--lower_case False --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device}



