base_directory=/media/saadi/Saadi12/ #/home/hsaadi/helper_folder
model=bert
lang=de
prefix=mbert
batch_size=1
max_sent=1
max_len=256
device=cpu
lower_case=False
model_path=bert-base-multilingual-cased
total_sentences=2000000
wiki=${lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt
lm_path=${base_directory}/helper_folder/saved_models/indic.bn-en.cao.pt
single_layer_path=${base_directory}/helper_folder/saved_models/mbert.transformer.bn.bn-de-en.pt
adapter_path=${base_directory}/helper_folder/saved_adapters/adapter_mbert_transformer_adapter_bn-de-en
adapter_name=adapter_mbert_transformer_adapter_bn-de-en

:'
python generate_avg_anchors.py --input_dict ${base_directory}/helper_folder/data/words/${lang}.txt \
--lang ${lang} --input_file  ${base_directory}/helper_folder/${wiki}  \
--output_file  ${base_directory}/helper_folder/embeddings/${prefix}.${lang} \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${model_path} --temp_file temp/temp.txt \
--lower_case False --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device} \
--single_layer ${single_layer_path} \
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name} \
#--single_layer ${single_layer_path}

lang=en
wiki=${lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt
 
python generate_avg_anchors.py --input_dict ${base_directory}/helper_folder/data/words/${lang}.txt \
--lang ${lang} --input_file  ${base_directory}/helper_folder/${wiki}  \
--output_file  ${base_directory}/helper_folder/embeddings/${prefix}.${lang} \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${model_path} --temp_file temp/temp.txt \
--lower_case False --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device} \
--single_layer ${single_layer_path} \
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name} \
#--single_layer ${single_layer_path}

lang=bn
wiki=${lang}wiki-20210501-pages-articles-multistream_preprocessed_filtered.txt


python generate_avg_anchors.py --input_dict ${base_directory}/helper_folder/data/words/${lang}.txt \
--lang ${lang} --input_file  ${base_directory}/helper_folder/${wiki}  \
--output_file  ${base_directory}/helper_folder/embeddings/${prefix}.${lang} \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${model_path} --temp_file temp/temp.txt \
--lower_case False --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device} \
--single_layer ${single_layer_path} \
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name} \
#--single_layer ${single_layer_path}

'
python generate_visualization.py --de_input_file de_sentences.txt \
--en_input_file en_sentences.txt \
--bn_input_file bn_sentences.txt \
--en_emb_file ${base_directory}/helper_folder/embeddings/${prefix}.en \
--bn_emb_file ${base_directory}/helper_folder/embeddings/${prefix}.bn \
--de_emb_file ${base_directory}/helper_folder/embeddings/${prefix}.de \
--max_sent ${max_sent} --max_len ${max_len} --model ${model} \
--model_path ${model_path} \
--lower_case ${lower_case} --total_sent ${total_sentences} --dimension 768 --batch ${batch_size} --device ${device} \
--single_layer ${single_layer_path} \
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name} \
#--single_layer ${single_layer_path}
#--de_emb_file ${base_directory}/helper_folder/embeddings/${prefix}.de \

