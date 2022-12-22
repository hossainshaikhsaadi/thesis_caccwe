
base_directory=/home/hsaadi/helper_folder
batch_size=4
epochs=3
prefix=xlm
xnli_lang=en
src_lang=en
lang_model=bn-de
model_name=xlm-roberta
lower_case=False
model_path=xlm-roberta-base
dimension=768
method=none
max_len=128
xnli_method=cls
device=cuda
train_mode=True
test_mode=True
#method_path=${base_directory}/saved_models/xlm.bn-de-en.cao.pt #${base_directory}/saved_models/${prefix}.${lang_model}-en.pt
xnli_path=${base_directory}/saved_models/${method}.${prefix}.${xnli_lang}.transformer.pt
#finetuned_lm_path=${base_directory}/saved_models/${method}.${prefix}.${xnli_lang}.lm.pt
mapping_method=${base_directory}/mapping_matrix/dbert.none.de-en.vecmap.npy #vecmap.npy
mapping_matrix_name=mbert.xnli.${src_lang}-en.${mapping_method}
mapping_matrix_path=${base_directory}/mapping_matrix/${mapping_matrix_name}
log_file=${base_directory}/log_files/${prefix}.${method}.${src_lang}.none.txt
single_layer_type=transformer
finetuned_lm_path=${base_directory}/saved_models/${method}.${single_layer_type}.${prefix}.${xnli_lang}.lm.pt
single_layer_path=${base_directory}/saved_models/${prefix}.${single_layer_type}.bn-de-en.pt
adapter_path=${base_directory}/saved_adapters/
adapter_name=adapter_${prefix}_${single_layer_type}_adapter_bn-de-en

python xnli_classification.py \
--train_file ${base_directory}/data/xnli_data/xnli_clean/clean_${src_lang}_train.txt \
--dev_file ${base_directory}/data/xnli_data/xnli_clean/clean_${src_lang}_eval.txt \
--test_file ${base_directory}/data/xnli_data/xnli_clean/clean_${src_lang}_test.txt \
--xnli_method ${xnli_method} \
--xnli_path ${xnli_path} \
--batch_size ${batch_size} \
--lower_case ${lower_case} \
--device ${device} \
--epochs ${epochs} \
--log_file ${log_file} \
--model_name ${model_name} \
--model_path ${model_path} \
--dimension ${dimension} \
--max_len ${max_len} \
--method ${method} \
--finetuned_lm_path ${finetuned_lm_path} \
--single_layer ${single_layer_path}
#--mapping_matrix ${mapping_method}
#--single_layer ${single_layer_path} \
#--adapter_path ${adapter_path} \
#--adapter_name ${adapter_name}
#--mapping_matrix ${mapping_matrix_path}
#--single_layer ${single_layer_path}
#--method_path ${method_path} \
