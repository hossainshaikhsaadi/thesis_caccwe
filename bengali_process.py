from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize  


bengali_file_read = open('./data/language_corpus/original_corpus.bn', 'r')
bengali_file_write = open('./data/language_corpus/original_corpus.bn.tokenized', 'w')

remove_nuktas=False
factory=IndicNormalizerFactory()
normalizer=factory.get_normalizer("bn")

for line in bengali_file_read:
    line = line.strip()
    normalized_line = normalizer.normalize(line)
    tokenized_line = indic_tokenize.trivial_tokenize(normalized_line)
    sentence = " ".join(tokenized_line) + "\n"
    bengali_file_write.write(sentence)

bengali_file_write.close()
bengali_file_read.close()

