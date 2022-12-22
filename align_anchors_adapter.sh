single_layer_type=transformer
cd vecmap


src_lang=de
tgt_lang=en
prefix=mbert
lang_pair=${src_lang}-${tgt_lang}

python map_embeddings.py /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.${src_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.${tgt_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${src_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${tgt_lang} \
-d /home/hsaadi/helper_folder/data/dictionaries/clean_train_dicts/${lang_pair}.txt --normalize unit center --orthogonal \
--mapping_file /home/hsaadi/helper_folder/mapping_matrix/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.npy \
> /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.train 

python eval_translation.py /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${src_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${tgt_lang} \
-d /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
> /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.eval

src_lang=bn
tgt_lang=en
prefix=mbert
lang_pair=${src_lang}-${tgt_lang}

python map_embeddings.py /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.${src_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.${tgt_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${src_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${tgt_lang} \
-d /home/hsaadi/helper_folder/data/dictionaries/clean_train_dicts/${lang_pair}.txt --normalize unit center --orthogonal \
--mapping_file /home/hsaadi/helper_folder/mapping_matrix/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.npy \
> /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.train 

python eval_translation.py /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${src_lang} \
/home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${tgt_lang} \
-d /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
> /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.eval


cd .. 

cd rcsls


src_lang=de
tgt_lang=en
prefix=mbert
lang_pair=${src_lang}-${tgt_lang}

python align.py --src_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.${src_lang} \
--tgt_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.${tgt_lang} \
--output_src /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${src_lang} \
--output_tgt /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${tgt_lang} \
--dico_test /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
--dico_train /home/hsaadi/helper_folder/data/dictionaries/clean_train_dicts/${lang_pair}.txt \
--mapping_file /home/hsaadi/helper_folder/mapping_matrix/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls --center \
> /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.train 

python eval.py --src_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${src_lang} \
--tgt_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${tgt_lang} \
--dico_test /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
--src_mat /home/hsaadi/helper_folder/mapping_matrix/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls-mat --center \
> /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.eval


src_lang=bn
tgt_lang=en
prefix=mbert
lang_pair=${src_lang}-${tgt_lang}

python align.py --src_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.${src_lang} \
--tgt_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.${tgt_lang} \
--output_src /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${src_lang} \
--output_tgt /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${tgt_lang} \
--dico_test /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
--dico_train /home/hsaadi/helper_folder/data/dictionaries/clean_train_dicts/${lang_pair}.txt \
--mapping_file /home/hsaadi/helper_folder/mapping_matrix/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls --center \
> /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.train 

python eval.py --src_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${src_lang} \
--tgt_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${tgt_lang} \
--dico_test /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
--src_mat /home/hsaadi/helper_folder/mapping_matrix/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls-mat --center \
> /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.eval

cd ..


src_lang=de
tgt_lang=en
prefix=mbert
lang_pair=${src_lang}-${tgt_lang}

python bli_evaluation.py --src_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${src_lang} \
--tgt_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${tgt_lang} \
--dict /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
>  /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls

python bli_evaluation.py --src_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${src_lang} \
--tgt_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${tgt_lang} \
--dict /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
>/home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap


src_lang=bn
tgt_lang=en
prefix=mbert
lang_pair=${src_lang}-${tgt_lang}

python bli_evaluation.py --src_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${src_lang} \
--tgt_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls.${tgt_lang} \
--dict /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
>  /home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.rcsls

python bli_evaluation.py --src_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${src_lang} \
--tgt_emb /home/hsaadi/helper_folder/embeddings/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap.${tgt_lang} \
--dict /home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/${lang_pair}.txt \
>/home/hsaadi/helper_folder/results/bli/${prefix}.${single_layer_type}.adapter.${lang_pair}.vecmap



