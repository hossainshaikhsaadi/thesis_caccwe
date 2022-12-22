import pandas as pd
import jsonlines
import csv

def preprocess_bangla_xnli(directory, clean_directory, file_names):
    for file_index in range(len(file_names)):
        file_name = file_names[file_index]
        read_file_path = directory+"/"+file_name
        write_file_path = clean_directory+"/"+"clean_bn_"+file_name
        file_to_read = open(read_file_path,'r')
        file_to_write = open(write_file_path, 'w')
        for line in file_to_read:
            line = line.strip()
            lines = line.split("\t")
            if len(lines) == 3:
                sentence = lines[0].strip()+"|||"+lines[1].strip()+"|||"+lines[2].strip()+"\n"
                file_to_write.write(sentence)
        file_to_write.close()
        file_to_read.close()

def preprocess_original_xnli(directory, clean_directory, file_names, languages):
    for file_index in range(len(file_names)):
        file_name = file_names[file_index]
        read_file_path = directory+"/"+file_name
        xnli_dataframe = pd.read_csv(read_file_path, sep = '\t')
        for lang in languages:
            if "dev" in read_file_path :   
                write_file_path = clean_directory+"/"+"clean_"+lang.strip()+"_eval.txt"
            else:
                write_file_path = clean_directory+"/"+"clean_"+lang.strip()+"_test.txt"
            partial_xnli_dataframe = xnli_dataframe[['language','gold_label','sentence1_tokenized','sentence2_tokenized']]
            xnli_lang_dataframe = partial_xnli_dataframe[partial_xnli_dataframe.language == lang]
            rows = xnli_lang_dataframe.shape[0]
            cols = xnli_lang_dataframe.shape[1]
            file_to_write = open(write_file_path, 'w')
            for i in range(rows):
                label = xnli_lang_dataframe.iloc[i,1].strip()
                sentence1 = xnli_lang_dataframe.iloc[i,2].strip()
                sentence2 = xnli_lang_dataframe.iloc[i,3].strip()
                sentence = sentence1+"|||"+sentence2+"|||"+label+"\n"
                file_to_write.write(sentence)            
            file_to_write.close()

def preprocess_original_en_mnli(directory, clean_directory, file_name):
    read_file_path = directory+"/"+file_name
    write_file_path = clean_directory+"/"+"clean_en_train.txt"
    file_to_write = open(write_file_path, 'w')
    data = jsonlines.open(read_file_path)
    for line_data in data.iter():
        label = line_data['gold_label']
        sentence1 = line_data['sentence1']
        sentence2 = line_data['sentence2']
        sentence = sentence1+"|||"+sentence2+"|||"+label+"\n"
        file_to_write.write(sentence)  
    file_to_write.close()

def preprocess_original_mt_mnli(directory, clean_directory, file_name, lang):
    read_file_path = directory+"/"+file_name
    file_to_read = open(read_file_path)
    write_file_path = clean_directory+"/"+"clean_"+lang.strip()+"_train.txt"
    file_to_write = open(write_file_path, 'w')
    mnli_data = csv.reader(file_to_read, delimiter = '\t') 
    for line in mnli_data:
        if len(line) == 3:
            sentence = line[0].strip()+"|||"+line[1].strip()+"|||"+line[2].strip()+"\n"
            file_to_write.write(sentence)
    file_to_read.close()
    file_to_write.close()   
    
xnli_original_train_file = "xnli.dev.tsv"
xnli_original_test_file = "xnli.test.tsv"
xnli_bangla_train_file = "train.txt"
xnli_bangla_test_file = "test.txt"
xnli_bangla_eval_file = "eval.txt"
mnli_original_en_train_file = "multinli_1.0_train.jsonl"
mnli_original_de_train_file = "multinli.train.de.tsv"

base_directory = "/home/hsaadi/helper_folder/data/xnli_data/"
xnli_directory = base_directory + "xnli"
en_mnli_directory =  base_directory + "xnli_en"
mt_mnli_directory =  base_directory + "xnli_mt"
bangla_xnli_directory =  base_directory + "xnli_bn"

clean_directory = "/home/hsaadi/helper_folder/data/xnli_data/xnli_clean"

bangla_files = [xnli_bangla_train_file, xnli_bangla_test_file, xnli_bangla_eval_file]
xnli_files = [xnli_original_train_file, xnli_original_test_file]
languages = ['de', 'en']
language = "de"

preprocess_bangla_xnli(bangla_xnli_directory, clean_directory, bangla_files)
preprocess_original_xnli(xnli_directory, clean_directory, xnli_files, languages)
preprocess_original_en_mnli(en_mnli_directory, clean_directory, mnli_original_en_train_file)
preprocess_original_mt_mnli(mt_mnli_directory, clean_directory, mnli_original_de_train_file, language)



