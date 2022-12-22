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
import random

def read_file(path):
    words = []
    sentences = []
    sent_file = open(path, 'r')
    for sent in sent_file:
        tokens = sent.strip().split()
        word = tokens[-1]
        sentence = ' '.join([str(i) for i in tokens[0:len(tokens)-1]])
        words.append(word)
        sentences.append(sentence)
    print(sentences)
    return words, sentences

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

def load_lm(language_model):
    state_dict_1 = torch.load(args.lm_path, map_location=torch.device(args.device))['state_dict']
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
        self.linear_layer_1 = nn.Linear(768, 768)
        if use_cuda == True:
            self.cuda()
    
    def forward(self, data):
        x = self.linear_layer_1(data)
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

def get_single_layer():
    if args.device == 'cuda':
        use_gpu = True
    if 'linear' in args.single_layer:
        layer = Single_Linear(use_gpu)
        layer.load_state_dict(torch.load(args.single_layer))
    else: 
        layer = Single_Transformer_Encoder(use_gpu)
        layer.load_state_dict(torch.load(args.single_layer))
    return layer

def get_adapters(model):
    model.load_adapter(args.adapter_path)
    model.set_active_adapters(args.adapter_name)
    return model

def get_embeddings(path):
    emb_file = open(path, 'r')
    embeddings = []
    i = 0
    for line in emb_file:
        if i == 0:
            i = i + 1
            continue
        line = line.strip().split()
        array = [float(i) for i in line[1:len(line)]]    
        embeddings.append(array)
    return embeddings

def get_adapters(model):
    model.load_adapter(args.adapter_path)
    model.set_active_adapters(args.adapter_name)
    #model.train_adapter(args.adapter_name)
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
    parser.add_argument('--de_input_file', type = str, default = None, help = 'path to sentence file')
    parser.add_argument('--en_input_file', type = str, help = 'path to sentence file')
    parser.add_argument('--bn_input_file', type = str, help = 'path to sentence file')
    parser.add_argument('--de_emb_file', type = str, default = None, help = 'path to sentence file')
    parser.add_argument('--en_emb_file', type = str, help = 'path to sentence file')
    parser.add_argument('--bn_emb_file', type = str, help = 'path to sentence file')
    parser.add_argument('--max_sent', type = int, help = 'number of senteces used for anchor generation')
    parser.add_argument('--max_len', type = int, help = 'maximum length of a sentence for consideration')
    parser.add_argument('--model', type = str, help = 'model name bert/xlm')
    parser.add_argument('--model_path', type = str, help = 'model or path')
    parser.add_argument('--lm_path', type = str, default = None, help = 'path to the finetuned language_model')
    parser.add_argument('--dimension', type = int, help = 'dimension of embeddings')
    parser.add_argument('--lower_case', type = bool, default = False, help = 'do lower case')
    parser.add_argument('--total_sent', type = int, help = 'number of total sentences')
    parser.add_argument('--batch', type = int, help = 'batch size')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'cpu or cuda (if available')
    parser.add_argument('--single_layer', type = str, default = None, help = 'path to the linear or transformer layer')
    parser.add_argument('--adapter_path', type = str, default = None, help = 'path to the adapter')
    parser.add_argument('--adapter_name', type = str, default = None, help = 'name of the adapter')
    args = parser.parse_args()

    if args.de_input_file != None:
        de_words, de_sentences = read_file(args.de_input_file)
    en_words, en_sentences = read_file(args.en_input_file)
    bn_words, bn_sentences = read_file(args.bn_input_file)
    all_words = []
    if args.de_input_file != None:
        all_words.extend(de_words)
    all_words.extend(en_words)
    all_words.extend(bn_words)

    if args.de_emb_file != None:
        de_embeddings = get_embeddings(args.de_emb_file)
    en_embeddings = get_embeddings(args.en_emb_file)
    bn_embeddings = get_embeddings(args.bn_emb_file)

    
    title = None
    if args.model == "bert": 
        tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = BertModel.from_pretrained(args.model_path)
        title = "mBERT with no finetuning"
        if args.lm_path != None:
            title = "mBERT fintuned"
        elif  args.single_layer != None and 'adapter' in args.single_layer:
            title = "mBERT with a trained adapter and transformer layer on top"
        elif args.single_layer != None and 'linear' in args.single_layer:
            title = "mBERT with a trained linear layer on top"
        elif args.single_layer != None and 'transformer' in args.single_layer:
            title = "mBERT with a trained transformer encoder layer on top"
    if args.model == "distilbert": 
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = DistilBertModel.from_pretrained(args.model_path)
        title = "dBERT with no finetuning"
        if args.lm_path != None:
            title = "dBERT fintuned"
        elif args.single_layer != None and 'linear' in args.single_layer:
            title = "dBERT with a trained linear layer on top"
        elif args.single_layer != None and 'transformer' in args.single_layer:
            title = "dBERT with a trained transformer encoder layer on top"
    if args.model == "albert": 
        tokenizer = AlbertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = AlbertModel.from_pretrained(args.model_path)
        title = "indic-bert with no finetuning"
        if args.lm_path != None:
            title = "indic-bert fintuned"
        elif args.single_layer != None and 'linear' in args.single_layer:
            title = "indic-bert with a trained linear layer on top"
        elif args.single_layer != None and 'transformer' in args.single_layer:
            title = "indic-bert with a trained transformer encoder layer on top"
    if args.model == "electra": 
        tokenizer = ElectraTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = ElectraModel.from_pretrained(args.model_path)
    if args.model == "xlm": 
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = XLMRobertaModel.from_pretrained(args.model_path)
        title = "XLM-RoBERTa with no finetuning"
        if args.lm_path != None:
            title = "XLM-RoBERTa fintuned"
        elif args.single_layer != None and 'linear' in args.single_layer:
            title = "XLM-RoBERTa with a trained linear layer on top"
        elif args.single_layer != None and 'transformer' in args.single_layer:
            title = "XLM-RoBERTa with a trained transformer encoder layer on top"

    single_layer = None
    if args.lm_path != None:
        model = load_lm(model)
    if args.single_layer != None:
        single_layer = get_single_layer()
    if args.adapter_path != None:
        model = get_adapters(model)
        model = prepare_model(model)

    model.eval()
    model.to(args.device)

    
    if args.de_emb_file != None:
        de_anchors = []
        for index, keyword in enumerate(de_words):
            anchor = get_anchor(keyword, [de_sentences[index]], args.batch, args.device)
            de_anchors.append(anchor)
    

    en_anchors = []
    for index, keyword in enumerate(en_words):
        anchor = get_anchor(keyword, [en_sentences[index]], args.batch, args.device)
        en_anchors.append(anchor)

    bn_anchors = []
    for index, keyword in enumerate(bn_words):
        anchor = get_anchor(keyword, [bn_sentences[index]], args.batch, args.device)
        bn_anchors.append(anchor)

    all_anchors = []
    if args.de_emb_file != None:
        all_anchors.extend(de_anchors)
    all_anchors.extend(en_anchors)
    all_anchors.extend(bn_anchors)
    
    
    all_embeddings = []
    if args.de_emb_file != None:
        all_embeddings.extend(de_embeddings)
    all_embeddings.extend(en_embeddings)
    all_embeddings.extend(bn_embeddings)
    random.shuffle(all_embeddings)
    start = len(all_embeddings)
    all_embeddings.extend(all_anchors)
    end = len(all_embeddings)
    emb = np.array(all_embeddings)

    #from sklearn.manifold import TSNE
    #tsne_model = TSNE(n_components=2, random_state=0)
    #transformed_emb = tsne_model.fit_transform(emb) 
    #new_anchors_tsne = transformed_emb[start:end]
    
    
    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    #all_embeddings = min_max_scaler.fit_transform(all_embeddings)
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)
    new_anchors_pca  = pca.transform(emb[start:end]) 
    

    #from sklearn import preprocessing

    #min_max_scaler = preprocessing.MinMaxScaler()
    #new_anchors_tsne = min_max_scaler.fit_transform(new_anchors_tsne)
    new_anchors_pca = min_max_scaler.fit_transform(new_anchors_pca)


    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    prop = fm.FontProperties(fname='Siyamrupali.ttf')
    '''
    fig, ax = plt.subplots()
    tsne_title = title + "- TSNE"
    ax.set_title(tsne_title)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.scatter(new_anchors_tsne[:,0], new_anchors_tsne[:,1], c = 'red')
    for i, txt in enumerate(all_words):
        ax.annotate(txt, (new_anchors_tsne[i,0], new_anchors_tsne[i,1]), fontproperties=prop)
    plt.savefig(tsne_title, bbox_inches = "tight")
    plt.show()
    '''
    fig, ax = plt.subplots()
    pca_title = title
    ax.set_title(pca_title)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    print(new_anchors_pca)
    ax.scatter(new_anchors_pca[0:5,0], new_anchors_pca[0:5,1], c = 'green')
    ax.scatter(new_anchors_pca[5:10,0], new_anchors_pca[5:10,1], c = 'red')
    ax.scatter(new_anchors_pca[10:15,0], new_anchors_pca[10:15,1], c = 'blue')
    ax.legend(["German", "English","Bengali"])
    for i, txt in enumerate(all_words):
        ax.annotate(txt, (new_anchors_pca[i,0], new_anchors_pca[i,1]), fontproperties=prop)
    plt.savefig(pca_title, bbox_inches = "tight")
    #plt.savefig(t, bbox_inches = "tight")
    plt.show()
    





    
    
     