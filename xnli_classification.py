
from transformers import BertTokenizer, BertModel
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AlbertTokenizer, AlbertModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import AdapterType, AdapterConfig
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import logging as log

import sys

from generate_avg_anchors import prepare_model
#sys.dont_write_bytecode = 1
#CUDA_LAUNCH_BLOCKING = 1


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class xnli_data(Dataset):
    def __init__(self, data, max_len, tokenizer, model_name):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.mode_name = model_name
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_data = self.data[idx]
        first_sentence = input_data[0].strip()
        second_sentence = input_data[1].strip()
        label = input_data[2].strip()
        token_ids = np.zeros((1, 512), dtype = int)
        token_masks = np.zeros((1, 512), dtype = int)
        token_type_ids = np.zeros((1, 512), dtype = int)
        if self.mode_name == 'xlm':
            first_sentence = "<s> " + first_sentence + " </s> </s> "
            second_sentence = second_sentence + " </s>"
        else:
            first_sentence = "[CLS] " + first_sentence + " [SEP] "
            second_sentence = second_sentence + " [SEP]"
        
        tokenized_first_sentence = self.tokenizer.tokenize(first_sentence)
        tokenized_second_sentence = self.tokenizer.tokenize(second_sentence)
        tokenized_input_sentence = tokenized_first_sentence + tokenized_second_sentence
        token_types= [0] * len(tokenized_first_sentence)
        token_type_ids[:, 0:len(tokenized_first_sentence)] = token_types
        token_type_ids[:, len(tokenized_first_sentence):len(tokenized_first_sentence)+len(tokenized_second_sentence)] = 1
        tokenized_input_sentence_ids = self.tokenizer.convert_tokens_to_ids(tokenized_input_sentence)
        token_ids[:, 0:len(tokenized_input_sentence_ids)] = tokenized_input_sentence_ids
        token_masks[:, 0:len(tokenized_input_sentence_ids)] = 1
        targets = ['neutral', 'contradiction', 'entailment']
        label = targets.index(label)
        return token_ids, token_masks, token_type_ids, label

class xnli_classifier(nn.Module):
    def __init__(self, l1_input, classes):
        super(xnli_classifier, self).__init__()
        self.linear_layer1 = nn.Linear(l1_input, classes)

    def forward(self, data, mode):
        x = F.dropout(data, p=0.10, training = mode)
        x = self.linear_layer1(data)
        output = F.softmax(x, dim =1)
        return output

def train_xnli_classifier():
    validation_loss = 1000
    for epoch in range(args.epochs):
        running_loss = 0
        total_data = 0
        correctly_classified = 0
        model.train()
        xnli_model.train()
        for iteration, batch_data in enumerate(train_dataloader):

            schedule_learning_rate(iteration, epoch, learning_rate)
            
            input_ids, input_masks, input_types, labels = batch_data
            input_ids = input_ids.squeeze(1).to(args.device)
            input_masks = input_masks.squeeze(1).to(args.device)
            input_types = input_types.squeeze(1).to(args.device)
            labels = torch.tensor([int(t) for t in labels]).to(args.device)
            

            if args.xnli_method == 'cls':
                if args.model_name == 'distilbert':
                    features = model(input_ids = input_ids, attention_mask = input_masks, 
                                    output_hidden_states = False)[0][:, 0, :]
                elif args.model_name == 'xlm-roberta':
                    features = model(input_ids = input_ids, attention_mask = input_masks, output_hidden_states = False)[0][:, 0, :]
                else:
                    features = model(input_ids = input_ids, attention_mask = input_masks, 
                                token_type_ids = input_types, output_hidden_states = False)[0][:, 0, :]
            elif args.xnli_method == 'avg':
                features = model(input_ids, input_masks, input_types, output_hidden_states = False)[0]
                averaged_features = torch.zeros(input_masks.shape[0], args.dimension, dtype = torch.float32)
                for n in range(input_masks.shape[0]):
                    limit = torch.nonzero(input_masks[0], as_tuple=True)[0].shape[0]
                    feature = features[n, 0:limit, :]
                    feature_mean = torch.mean(feature, dim = 0)
                    averaged_features[n, 0:args.dimension] = feature_mean[0:args.dimension]
                features = averaged_features
            
            if args.single_layer != None:
                features = features.unsqueeze(0)
                features = single_layer(features)
                features = features.squeeze(0)

            output = xnli_model(features, True)

            
            loss = F.cross_entropy(output, labels)

            optimizer_xnli.zero_grad()
            optimizer_lm.zero_grad()
            loss.backward()
            optimizer_xnli.step()
            optimizer_lm.step()

            _, predicted = torch.max(output, 1)
            correctly_classified = correctly_classified+ (predicted == labels).sum().item()

            running_loss = running_loss + loss.item()
            total_data = total_data + labels.size(0)

            if (iteration+1) % 100 == 0 :
                formatted_loss = "{0:.6f}".format(loss)
                formatted_running_loss = "{0:.6f}".format(running_loss)
                log_sentence = "EPOCH : "+str(epoch+1)+" ITERATION : "+str(iteration+1)
                log_sentence = log_sentence + " Training Running Loss : "+str(formatted_running_loss)
                log_sentence = log_sentence + " Training Batch Loss : "+str(formatted_loss)
                log.info(log_sentence)
            #if (iteration+1) == 200:
            #    new_validation_loss = validate_or_test_xnli_classifier(validation_dataloader, epoch, mode = 'train')

        training_epoch_accuracy = (correctly_classified/total_data)*100
        formatted_training_epoch_accuracy = "{0:.2f}".format(training_epoch_accuracy)
        training_epoch_loss = running_loss/total_data
        formatted_training_epoch_loss = "{0:.6f}".format(training_epoch_loss)
        log_sentence = "EPOCH : "+str(epoch+1)+" Training Epoch Loss : "+str(formatted_training_epoch_loss)
        log.info(log_sentence)
        log_sentence = "EPOCH : "+str(epoch+1)+" Training Epoch Accuracy : "+str(formatted_training_epoch_accuracy)
        log.info(log_sentence)
        new_validation_loss = validate_or_test_xnli_classifier(validation_dataloader, epoch, mode = 'train')
        #if new_validation_loss < validation_loss:
        save_xnli_model()
        save_finetuned_lm()


def validate_or_test_xnli_classifier(dataloader, epoch = 0, mode = 'train'):
    model.eval()
    xnli_model.eval()
    running_loss = 0
    total_data = 0
    correctly_classified = 0
    for iteration, batch_data in enumerate(dataloader):

        input_ids, input_masks, input_types, labels = batch_data
        input_ids = input_ids.squeeze(1).to(args.device)
        input_masks = input_masks.squeeze(1).to(args.device)
        input_types = input_types.squeeze(1).to(args.device)
        labels = torch.tensor([int(t) for t in labels]).to(args.device)

        if args.xnli_method == 'cls':
            if args.model_name == 'distilbert':
                features = model(input_ids = input_ids, attention_mask = input_masks, 
                                output_hidden_states = False)[0][:, 0, :]
            elif args.model_name == 'xlm-roberta':
                features = model(input_ids = input_ids, attention_mask = input_masks, output_hidden_states = False)[0][:, 0, :]
            else:
                features = model(input_ids = input_ids, attention_mask = input_masks, 
                                token_type_ids = input_types, output_hidden_states = False)[0][:, 0, :]
                    
            #if args.single_layer != None:
            #    features = single_layer(features)
            #    features = features[:, 0, :]
        elif args.xnli_method == 'avg':
            features = model(input_ids, input_masks, input_types, output_hidden_states = False)[0]
            if args.single_layer != None:
                features = single_layer(features)
            averaged_features = torch.zeros(input_masks.shape[0], args.dimension, dtype = torch.float32)
            for n in range(input_masks.shape[0]):
                limit = torch.nonzero(input_masks[0], as_tuple=True)[0].shape[0]
                feature = features[n, 0:limit, :]
                feature_mean = torch.mean(feature, dim = 0)
                averaged_features[n, 0:args.dimension] = feature_mean[0:args.dimension]
            features = averaged_features.to(args.device)
        
        if args.mapping_matrix != None:
            features = features.detach().cpu().numpy()
            features = mean_center(features)
            features = length_normalize(features)
            features = torch.Tensor(features).to(args.device)
            features = mapping_layer(features)

        if args.single_layer != None:
            features = features.unsqueeze(0)
            features = single_layer(features)
            features = features.squeeze(0)

        output = xnli_model(features, False)

        loss = F.cross_entropy(output, labels)

        _, predicted = torch.max(output, 1)
        correctly_classified = correctly_classified+ (predicted == labels).sum().item()

        running_loss = running_loss + loss.item()
        total_data = total_data + labels.size(0)

        if (iteration+1) % 100 == 0 :
            formatted_loss = "{0:.6f}".format(loss)
            formatted_running_loss = "{0:.6f}".format(running_loss)
            log_sentence = "EPOCH : "+str(epoch+1)+" ITERATION : "+str(iteration+1)
            if mode == 'train':
                log_sentence = log_sentence + " Validation Running Loss : "+str(formatted_running_loss)
                log_sentence = log_sentence + " Validation Batch Loss : "+str(formatted_loss)
            else:
                log_sentence = log_sentence + " Test Running Loss : "+str(formatted_running_loss)
                log_sentence = log_sentence + " Test Batch Loss : "+str(formatted_loss)
            log.info(log_sentence)
        #if (iteration+1) == 50:
        #    break
    accuracy = (correctly_classified/total_data)*100
    formatted_accuracy = "{0:.2f}".format(accuracy)
    loss = running_loss/total_data
    formatted_loss = "{0:.6f}".format(loss)
    if mode == 'train':
        log_sentence = "EPOCH : "+str(epoch+1)+" Validation Loss : "+str(formatted_loss)
        log.info(log_sentence)
        log_sentence = "EPOCH : "+str(epoch+1)+" Validation Accuracy : "+str(formatted_accuracy)
        log.info(log_sentence)
        return loss
    else:
        log_sentence = "EPOCH : "+str(epoch+1)+" Test Loss : "+str(formatted_loss)
        log.info(log_sentence)
        log_sentence = "EPOCH : "+str(epoch+1)+" Test Accuracy : "+str(formatted_accuracy)
        log.info(log_sentence)
        return 0

def save_xnli_model():
    model_dict = xnli_model.state_dict()
    torch.save(model_dict, args.xnli_path)

def save_finetuned_lm():
    model_dict = model.state_dict()
    torch.save({'state_dict' : model_dict}, args.finetuned_lm_path)


def load_xnli_model(x_model):
    x_model.load_state_dict(torch.load(args.xnli_path))
    return x_model

def provide_dataloader(file_name, workers, shuffle):
    data_list = read_data_files(file_name, args.max_len)
    data = xnli_data(data_list, args.max_len, tokenizer, args.model_name)
    dataloader = DataLoader(data, batch_size = args.batch_size, shuffle = shuffle, num_workers = workers)
    return dataloader

def get_model_tokenizer():
    if args.model_name == "bert": 
        tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = BertModel.from_pretrained(args.model_path)
    if args.model_name == "distilbert": 
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = DistilBertModel.from_pretrained(args.model_path)
    if args.model_name == "albert": 
        tokenizer = AlbertTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = AlbertModel.from_pretrained(args.model_path)
    if args.model_name == "electra": 
        tokenizer = ElectraTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = ElectraModel.from_pretrained(args.model_path)
    if args.model_name == "xlm-roberta": 
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_path, do_lower_case = args.lower_case)
        model = XLMRobertaModel.from_pretrained(args.model_path)
    return model , tokenizer

def read_data_files(file_path, max_len):
    file_to_read = open(file_path, 'r')
    data_list = []
    for line in file_to_read:
        lines = line.strip().split("|||")
        if len(lines[0].split()) > max_len - 2:
            continue 
        if len(lines[1].split()) > max_len - 2:
            continue 
        data_list.append(lines)
    return data_list

def load_lm(language_model):
    state_dict_1 = torch.load(args.method_path)['state_dict']
    state_dict_2 = {}
    for k, v in state_dict_1.items():
        if 'bert' in k:
            k = k.split(".")[1:]
            k = ".".join(k)
        state_dict_2[k] = v
    language_model.load_state_dict(state_dict_2)
    return language_model

def load_finetuned_lm(language_model):
    state_dict_1 = torch.load(args.finetuned_lm_path)['state_dict']
    state_dict_2 = {}
    for k, v in state_dict_1.items():
        state_dict_2[k] = v
    #if args.adapter_name != None:
    #    language_model = get_model_tokenizer()[0]
    #    language_model.load_state_dict(state_dict_2)
    #    language_model = get_adapters(language_model)
    #else:
    language_model.load_state_dict(state_dict_2)
    return language_model

def set_new_lr(rate):
    for param_group in optimizer_xnli.param_groups:
        param_group['lr'] = rate
    for param_group in optimizer_lm.param_groups:
        param_group['lr'] = rate
    
def schedule_learning_rate(iter, ep, lrate):
    if ep == 0:
        if (iter+1) <= number_of_steps_batch:
            warmup_co = learning_rate / number_of_steps_batch
            new_lr = (iter+1) * warmup_co
            set_new_lr(new_lr)

def get_mapping_matrix_layer():
    if "vecmap" in args.mapping_matrix:
        mapping_matrix = torch.tensor(np.load(args.mapping_matrix), dtype = torch.float32)
    else:
        matrix_file = open(args.mapping_matrix, 'r')
        mapping_matrix = np.zeros((args.dimension, args.dimension))
        for i, line in enumerate(matrix_file):
            if i == 0:
                continue
            line = line.strip().split()
            line_float = [float(d) for d in line]
            mapping_matrix[i-1, :] = line_float
        mapping_matrix = torch.tensor(mapping_matrix, dtype = torch.float32)
    mapping = nn.Linear(args.dimension, args.dimension)
    mapping.weight.data = mapping_matrix
    mapping.weight.requires_grad = False
    mapping.bias.data = torch.zeros(1, args.dimension, dtype = torch.float32)
    mapping.bias.requires_grad = False
    mapping.to(args.device)
    return mapping

def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]

def mean_center(matrix):
    avg = np.mean(matrix, axis=0)
    return matrix - avg

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

    
    def forward(self, data):
        x = self.transformer_encoder(data)
        return x

def get_single_layer():
    use_gpu = False
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



if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Arguments for cleaning dictionaries')
    parser.add_argument('--train_file', type = str, help = 'path to training file')
    parser.add_argument('--dev_file', type = str, help = 'path to validation file')
    parser.add_argument('--test_file', type = str, help = 'path to testing file')
    parser.add_argument('--train_mode', type = bool, default = False, help = 'train or test')
    parser.add_argument('--test_mode', type = bool, default = True, help = 'train or test')
    parser.add_argument('--xnli_method', type = str, default = 'cls', help = 'will use only cls or average of the token embeddings')
    parser.add_argument('--method', type = str, default = 'none', help = 'none or anchor or cao or linear or transformer or adapter')
    parser.add_argument('--method_path', type = str, default = None, help = 'path to the saved model')
    parser.add_argument('--xnli_path', type = str, help = 'path to the saved model or path for saving the model')
    parser.add_argument('--max_len', type = int, default = 64, help = 'maximum length of a sentence for consideration')
    parser.add_argument('--mapping_matrix', type = str, default = None, help = 'path to the mapping matrix')
    parser.add_argument('--model_name', type = str, help = 'model name bert/xlm')
    parser.add_argument('--model_path', type = str, help = 'model or path')
    parser.add_argument('--dimension', type = int, default = 768, help = 'dimension of embeddings')
    parser.add_argument('--lower_case', type = bool, default = False, help = 'do lower case')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'cpu or cuda (if available)')
    parser.add_argument('--epochs', type = int, default = 1, help = 'cpu or cuda (if available)')
    parser.add_argument('--log_file', type = str, default = 'sample.log', help = 'cpu or cuda (if available)')
    parser.add_argument('--finetuned_lm_path', type = str, default = None, help = 'cpu or cuda (if available)')
    parser.add_argument('--single_layer', type = str, default = None, help = 'path to the linear or transformer layer')
    parser.add_argument('--adapter_path', type = str, default = None, help = 'path to the adapter')
    parser.add_argument('--adapter_name', type = str, default = None, help = 'name of the adapter')

    args = parser.parse_args()

    log.basicConfig(format='%(asctime)s %(message)s', 
                datefmt='%m/%d/%Y %I:%M:%S %p', 
                level = log.INFO,
                filename = args.log_file,
                filemode = "w")
    
    workers = 1
    shuffle = True
    xnli_classifier_layer_1_input = 768
    xnli_classifier_classes = 3
    learning_rate = 0.000005
    test_mode = 'test'
    
    model, tokenizer = get_model_tokenizer()

    if args.method == 'cao':
        model = load_lm(model)
    if args.adapter_path != None:
        model = get_adapters(model)
        model = prepare_model(model)
    #initial_model = model
    
    if args.mapping_matrix != None:
        mapping_layer = get_mapping_matrix_layer()
    if args.single_layer != None:
        single_layer = get_single_layer()
        single_layer.to(args.device)
        single_layer.eval()
    
    if args.train_mode == True:
        train_dataloader = provide_dataloader(args.train_file, workers = workers, shuffle = shuffle)
        validation_dataloader = provide_dataloader(args.dev_file, workers = workers, shuffle = shuffle)
    test_dataloader = provide_dataloader(args.test_file, workers = workers, shuffle = shuffle)

    xnli_model = xnli_classifier(xnli_classifier_layer_1_input, 
                                xnli_classifier_classes)
    
    if args.train_mode == True:
        number_of_steps = int(0.1 * (len(train_dataloader) * args.batch_size))
        number_of_steps_batch = int(number_of_steps / args.batch_size)
    
    if args.train_mode == True:
        model.to(args.device)
        xnli_model.to(args.device)
        optimizer_xnli = torch.optim.Adam(list(xnli_model.parameters()), lr = learning_rate)
        optimizer_lm = torch.optim.Adam(list(model.parameters()), lr = learning_rate)
        train_xnli_classifier()
    if args.test_mode == True:
        xnli_model = load_xnli_model(xnli_model)
        model = load_finetuned_lm(model)
        model.to(args.device)
        xnli_model.to(args.device)
        validate_or_test_xnli_classifier(test_dataloader, mode = test_mode)
    