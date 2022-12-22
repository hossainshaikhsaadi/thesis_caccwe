

model=xlm
tgt_lang=en
prefix=xlm
lm_path=indic.bn-en.cao.pt

batch_size=5
max_sent=100
max_len=256
device=cuda
src_lower_case=True
tgt_lower_case=True
src_model_path=xlm-roberta-base
tgt_model_path=xlm-roberta-base
total_sentences=200000000
base_directory=/home/hsaadi/helper_folder
single_layer_type=transformer
single_layer_path=${base_directory}/saved_models/${prefix}.${single_layer_type}.bn-de-en.pt
adapter_path=${base_directory}/saved_adapters/
adapter_name=adapter_${prefix}_${single_layer_type}_adapter_bn-de-en

:'
src_lang=de
lang_pair=${src_lang}-${tgt_lang}
src_wiki=${src_lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt
tgt_wiki=${tgt_lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt


python generate_avg_anchors.py --input_dict /home/hsaadi/helper_folder/data/words/${src_lang}.txt \
--lang ${src_lang} --input_file  /home/hsaadi/helper_folder/${src_wiki}  \
--output_file  /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.${lang_pair}.${src_lang} \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${src_model_path} --temp_file temp/temp.txt \
--lower_case False --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device} \
--single_layer ${single_layer_path} 
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name} 
#--single_layer ${single_layer_path}
#--lm_path /home/hsaadi/helper_folder/saved_models/${lm_path}

'

previous_lang_pair=${lang_pair}
src_lang=bn
lang_pair=${src_lang}-${tgt_lang}
src_wiki=${src_lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt

python generate_avg_anchors.py --input_dict /home/hsaadi/helper_folder/data/words/${src_lang}.txt \
--lang ${src_lang} --input_file  /home/hsaadi/helper_folder/${src_wiki}  \
--output_file  /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.${lang_pair}.${src_lang} \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${src_model_path} --temp_file temp/temp.txt \
--lower_case False --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device} \
--single_layer ${single_layer_path} 
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name} 
#--single_layer ${single_layer_path}
#--lm_path /home/hsaadi/helper_folder/saved_models/${lm_path}

:'
tgt_wiki=${tgt_lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt
lang_pair=${src_lang}-${tgt_lang}


python generate_avg_anchors.py --input_dict /home/hsaadi/helper_folder/data/words/${tgt_lang}.txt \
--lang ${tgt_lang} --input_file  /home/hsaadi/helper_folder/${tgt_wiki}  \
--output_file  /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.${lang_pair}.${tgt_lang} \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${tgt_model_path} --temp_file temp/temp.txt \
--lower_case False --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device} \
--single_layer ${single_layer_path} 
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name} 
#--lm_path /home/hsaadi/helper_folder/saved_models/${lm_path}

cp /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.${lang_pair}.${tgt_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.${previous_lang_pair}.${tgt_lang}

#cp /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.${lang_pair}.${tgt_lang} \
#/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.${previous_lang_pair}.${tgt_lang}
'