import argparse
from transformers import BertTokenizer, BertModel
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AlbertTokenizer, AlbertModel
from transformers import ElectraTokenizer, ElectraModel
import torch
import numpy as np
from tqdm import tqdm
from flashtext import KeywordProcessor
import torch.nn as nn


def read_dict(path, language):
    dict_file = open(path, 'r')
    anchor_words = []
    for line in dict_file:
        line = line.strip()
        words = line.split()
        if language == 'bn' or language == 'de':
            anchor_words.append(words[0])
        else:
            anchor_words.append(words[1])
    return anchor_words

def read_word_file(path):
    words = []
    word_file = open(path, 'r')
    for word in word_file:
        word = word.strip()
        words.append(word)
    return words


def read_file(words, language, path, max_sent, max_len):
    input_file = open(path, 'r')
    data_dict = dict()
    keyword_processor = KeywordProcessor(case_sensitive=True)
    for word in words:
        word = word.strip()
        cap_word = word.capitalize()
        up_word = word.upper()
        word_1 = " "+word+" "
        cap_word = " "+cap_word+" "
        up_word = " "+up_word+" "
        keyword_processor.add_keyword(word_1, word)
        keyword_processor.add_keyword(cap_word, word)
        keyword_processor.add_keyword(up_word, word)
    count = 0    
    for line in tqdm(input_file):
        line = line.strip()
        if len(line.split()) > max_len:
            continue
        kws = keyword_processor.extract_keywords(line)
        for kw in kws:
            kw = kw.strip()
            if len(data_dict.setdefault(kw, [])) < max_sent:
                data_dict[kw].append(line)
        count = count + 1
        if count > args.total_sent:
            break
    return data_dict

def get_anchor(keyword, sentences, batch_size, device):
    
    anchor_sum = np.zeros((768, ))
    total_correct_count = 0
    for i in range(0, len(sentences), batch_size):
        if i+batch_size <= len(sentences):
            current_sentences = sentences[i:i+batch_size]
        else:
            current_sentences = sentences[i:len(sentences)]
        
        batch_token_ids_list = []
        batch_anchor_masks_list = []
        batch_correct_count = 0
        
        for index, sentence in enumerate(current_sentences):
            words = sentence.split(' ')

            tokens = []
            anchor_mask = []

            if args.model == "xlm":
                tokens.extend(['<s>'])
            else:
                tokens.extend(['[CLS]'])

            anchor_mask.append(0)
            unk_in_keyword = 0

            for word in words:
                word = word.strip()
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                if word == keyword or word == keyword.capitalize() or word == keyword.upper():
                    for _ in range(len(word_tokens)):
                        anchor_mask.append(1)
                else:
                    for _ in range(len(word_tokens)):
                        anchor_mask.append(0)
                #if '[UNK]' in word_tokens:
                #    unk_in_keyword = 1

            if args.model == "xlm":
                tokens.extend(['</s>'])
            else:
                tokens.extend(['[SEP]'])

            anchor_mask.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if len(input_ids) < 512 and unk_in_keyword == 0:
                batch_token_ids_list.append(input_ids)
                batch_anchor_masks_list.append(anchor_mask)
                batch_correct_count = batch_correct_count + 1
                total_correct_count = total_correct_count + 1
                
        batch_input_ids = np.zeros((batch_correct_count, 512), dtype = int)
        batch_attention_mask = np.zeros((batch_correct_count, 512), dtype = int)
        batch_anchor_mask = np.zeros((batch_correct_count, 512), dtype = int)
        
        if batch_correct_count == 0:
            continue
        
        for j in range(batch_correct_count): 
            input_ids = batch_token_ids_list[j]
            anchor_mask = batch_anchor_masks_list[j]
            batch_input_ids[j, 0:len(input_ids)] = input_ids
            batch_attention_mask[j, 0:len(input_ids)] = 1
            batch_anchor_mask[j, 0:len(anchor_mask)] = anchor_mask

    
        attention_mask_tensor = torch.tensor(batch_attention_mask).to(device)
        input_ids_tensor = torch.tensor(batch_input_ids).to(device)
        anchor_mask_tensor = torch.tensor(batch_anchor_mask).to(device)
        

        features = model(input_ids_tensor, attention_mask = attention_mask_tensor,
                                output_hidden_states = False, return_dict = True)[0]

        if args.single_layer != None:
            features = single_layer(features)

        for batch_index in range(features.shape[0]):
            try:
                sentence_features = features[batch_index]
                sentence_anchor_masks = anchor_mask_tensor[batch_index]
                sentence_anchor_indices = torch.nonzero(sentence_anchor_masks, as_tuple=True)[0]
                keyword_features = torch.index_select(sentence_features, 0, sentence_anchor_indices)
                keyword_features = keyword_features.to('cpu').detach().numpy()
                if keyword_features.shape[0] > 1:
                    keyword_sum = np.zeros((768, ))
                    for k in range(keyword_features.shape[0]):
                        keyword_sum = np.add(keyword_sum, keyword_features[k])
                    keyword_sum = np.divide(keyword_sum, keyword_features.shape[0])
                    anchor_sum = np.add(anchor_sum, keyword_sum)
                else:
                    anchor_sum = np.add(anchor_sum, keyword_features[0]) 
            except Exception as ex:
                total_correct_count = total_correct_count - 1
                continue     
            
    if total_correct_count > 0:
        anchor_numpy = np.divide(anchor_sum, total_correct_count)
        anchor = anchor_numpy.tolist()
        #print(len(anchor))
    else:
        anchor = []
    return anchor

def write_in_temp_file(word, anchor):
    sentence = word+" "+" ".join([str(i) for i in anchor])+"\n"
    temp_write_file.write(sentence)
    return 0

def prepare_embedding_file(dimension):
    temp_file = open(args.temp_file, 'r')
    output_file = open(args.output_file, 'w')
    sentence = str(total_emb)+" "+str(dimension)+"\n"
    output_file.write(sentence)
    for line in temp_file:
        line = line.strip()
        sentence = line + '\n'
        output_file.write(sentence)
    output_file.close()
    temp_file.close()
    return 0

def load_lm(language_model):
    state_dict_1 = torch.load(args.lm_path)['state_dict']
    state_dict_2 = {}
    for k, v in state_dict_1.items():
        if 'bert' in k:
            k = k.split(".")[1:]
            k = ".".join(k)
        state_dict_2[k] = v
    language_model.load_state_dict(state_dict_2)
    return language_model

class Single_Linear(nn.Module):
    def __init__(self, use_cuda = False):
        super(Single_Linear, self).__init__()
        self.linear_layer_1 = nn.Linear(768, 768, bias = True)
        if use_cuda == True:
            self.cuda()
    
    def forward(self, data):
        x = self.linear_layer_1(data)
        #x = self.linear_layer_2(x)
        return x

class Single_Transformer_Encoder(nn.Module):
    def __init__(self, use_cuda = False):
        super(Single_Transformer_Encoder, self).__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model = 768, nhead = 8)
        if use_cuda == True:
            self.cuda()
    
    def forward(self, data):
        x = self.transformer_encoder(data)
        return x

def get_adapters(model):
    path = args.adapter_path + "/"+ args.adapter_name 
    model.load_adapter(path)
    model.set_active_adapters(args.adapter_name)
    model.train_adapter(args.adapter_name)
    return model

def prepare_model(model):
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model

def get_single_layer():
    use_gpu = False
    if args.device == 'cuda':
        use_gpu = True
    if 'linear' in args.single_layer:
        layer = Single_Linear(use_gpu)
        layer.load_state_dict(torch.load(args.single_layer, map_location = torch.device(args.device)))
    else: 
        layer = Single_Transformer_Encoder(use_gpu)
        layer.load_state_dict(torch.load(args.single_layer, map_location = torch.device(args.device)))
    return layer
    
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Arguments for cleaning dictionaries')
    parser.add_argument('--input_dict', type = str, help = 'path to input dictionary or word file')
    parser.add_argument('--lang', type = str, help = 'language')
    parser.add_argument('--input_file', type = str, help = 'path to sentence file')
    parser.add_argument('--output_file', type = str, help = 'path to output file')
    parser.add_argument('--max_sent', type = int, help = 'number of senteces used for anchor generation')
    parser.add_argument('--max_len', type = int, help = 'maximum length of a sentence for consideration')
    parser.add_argument('--model', type = str, help = 'model name bert/xlm')
    parser.add_argument('--model_path', type = str, help = 'model or path')
    parser.add_argument('--lm_path', type = str, default = None, help = 'path to the finetuned language_model')
    parser.add_argument('--temp_file', type = str, help = 'tempprary file name or path')
    parser.add_argument('--dimension', type = int, help = 'dimension of embeddings')
    parser.add_argument('--lower_case', type = bool, default = False, help = 'do lower case')
    parser.add_argument('--total_sent', type = int, help = 'number of total sentences')
    parser.add_argument('--batch', type = int, help = 'batch size')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'cpu or cuda (if available')
    parser.add_argument('--single_layer', type = str, default = None, help = 'path to the linear or transformer layer')
    parser.add_argument('--adapter_path', type = str, default = None, help = 'path to the adapter')
    parser.add_argument('--adapter_name', type = str, default = None, help = 'name of the adapter')
    args = parser.parse_args()


    if args.model == "bert": 
        tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = BertModel.from_pretrained(args.model_path)
    if args.model == "distilbert": 
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = DistilBertModel.from_pretrained(args.model_path)
    if args.model == "albert": 
        tokenizer = AlbertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = AlbertModel.from_pretrained(args.model_path)
    if args.model == "electra": 
        tokenizer = ElectraTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = ElectraModel.from_pretrained(args.model_path)
    if args.model == "xlm": 
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = XLMRobertaModel.from_pretrained(args.model_path)

    single_layer = None
    if args.lm_path != None:
        model = load_lm(model)
    if args.single_layer != None:
        single_layer = get_single_layer()
    if args.adapter_path != None:
        model = get_adapters(model)
        #model = prepare_model(model)
    

    words = read_word_file(args.input_dict)
    data = read_file(words, args.lang, args.input_file, args.max_sent, args.max_len)


    model.eval()
    model.to(args.device)
    temp_write_file = open(args.temp_file, 'w')

    total_emb = 0
    for index, keyword in enumerate(data):
        sentences = data[keyword]
        sen_len = len(sentences)
        anchor = get_anchor(keyword, sentences, args.batch, args.device)
        if len(anchor) > 0:
            write_in_temp_file(keyword, anchor)
            total_emb = total_emb + 1
    temp_write_file.close()
    prepare_embedding_file(args.dimension)