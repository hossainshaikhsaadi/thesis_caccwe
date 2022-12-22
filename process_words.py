bn_train_dict = "/home/hsaadi/helper_folder/data/dictionaries/clean_train_dicts/bn-en.txt"
de_train_dict = "/home/hsaadi/helper_folder/data/dictionaries/clean_train_dicts/de-en.txt"

bn_test_dict = "/home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/bn-en.txt"
de_test_dict = "/home/hsaadi/helper_folder/data/dictionaries/clean_test_dicts/de-en.txt"

bn_words_path = "/home/hsaadi/helper_folder/data/words/bn.words.txt"
de_words_path = "/home/hsaadi/helper_folder/data/words/de.words.txt"
en_words_path = "/home/hsaadi/helper_folder/data/words/en.words.txt"

bn_words = []
de_words = []
en_words = []

count = 0
file_train_dict = open(bn_train_dict, 'r')
for line in file_train_dict:
    word = line.strip().split()[0].strip()
    bn_words.append(word)
file_train_dict.close()
file_test_dict = open(bn_test_dict, 'r')
for line in file_test_dict:
    word = line.strip().split()[0].strip()
    bn_words.append(word)
file_test_dict.close()
file_words = open(bn_words_path, 'r')
for word in file_words:
    word =  word.strip()
    bn_words.append(word)
file_words.close()

count = 0
file_train_dict = open(de_train_dict, 'r')
for line in file_train_dict:
    word = line.strip().split()[0].strip()
    de_words.append(word)
file_train_dict.close()
file_test_dict = open(de_test_dict, 'r')
for line in file_test_dict:
    word = line.strip().split()[0].strip()
    de_words.append(word)
file_test_dict.close()
file_words = open(de_words_path, 'r')
for word in file_words:
    word =  word.strip()
    de_words.append(word)
file_words.close()

count = 0
file_train_dict = open(de_train_dict, 'r')
for line in file_train_dict:
    word = line.strip().split()[1].strip()
    en_words.append(word)
file_train_dict.close()
file_test_dict = open(de_test_dict, 'r')
for line in file_test_dict:
    word = line.strip().split()[1].strip()
    en_words.append(word)
file_test_dict.close()

file_train_dict = open(bn_train_dict, 'r')
for line in file_train_dict:
    word = line.strip().split()[1].strip()
    en_words.append(word)
file_train_dict.close()
file_test_dict = open(bn_test_dict, 'r')
for line in file_test_dict:
    word = line.strip().split()[1].strip()
    en_words.append(word)
file_test_dict.close()        

file_words = open(en_words_path, 'r')
for word in file_words:
    word =  word.strip()
    en_words.append(word)
       
file_words.close()


new_bn_words_path = "/home/hsaadi/helper_folder/data/words/bn.txt"
new_de_words_path = "/home/hsaadi/helper_folder/data/words/de.txt"
new_en_words_path = "/home/hsaadi/helper_folder/data/words/en.txt"

file_write_bn = open(new_bn_words_path, 'w')
file_write_de = open(new_de_words_path, 'w')
file_write_en = open(new_en_words_path, 'w')

for word in bn_words:
    word = word.strip()+'\n'
    file_write_bn.write(word)
file_write_bn.close()

for word in de_words:
    word = word.strip()+'\n'
    file_write_de.write(word)
file_write_de.close()

for word in en_words:
    word = word.strip()+'\n'
    file_write_en.write(word)
file_write_en.close()

    



