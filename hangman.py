import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm
from torch.utils.data import Dataset
from torchvision import datasets
import random
from torch.utils.data import DataLoader
import math
import torch.nn as nn
import copy
from tqdm import tqdm
import torch.nn.functional as F




# class KLLossComputeMasked:
#     def __init__(self, generator, criterion):
#         self.generator = generator
#         self.criterion = criterion

#     def __call__(self, x, y, mask):
#         batch_size = x.shape[0]
#         x = self.generator(x, mask)
#         batch_loss = self.criterion(x, y)

#         loss = batch_loss / batch_size
#         return loss
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    query: shape (*, n_queries, d_k) n_queries is the maximum sentence length / max_sent_length - 1 if key from decoder
    key: (*, K, d_k) , K is the maximum sentence length / max_sent_length - 1 if key from decoder
    value: (*, K, d_v)

    scores: (n_quires, K)
    output: (n_queries, d_v)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -np.inf)
    p = F.softmax(scores, dim=-1)
    if dropout is not None:
        p = dropout(p)

    return torch.matmul(p, value), p


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.query_linear = nn.Linear(in_features=d_k * h,
                                      out_features=d_model,
                                      bias=False)
        self.key_linear = nn.Linear(in_features=d_k * h,
                                    out_features=d_model,
                                    bias=False)
        self.value_linear = nn.Linear(in_features=d_v * h,
                                      out_features=d_model,
                                      bias=False)

        self.attn = None  # not used for computation, only for visualization
        self.dropout = nn.Dropout(p=dropout)

        self.output_linear = nn.Linear(in_features=d_model,
                                       out_features=h * d_v)

    def forward(self, query, key, value, mask=None):
        """
        d_k * h = d_model

        query: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size,
        key: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size
        value: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size,

        output: shape (batch_size, max_sent_length, embedding_size)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        d_k = self.d_model // self.h

        n_batches = query.size(0)
        max_sent_length = query.size(1)
        query = self.query_linear(query).view(n_batches, max_sent_length, self.h, d_k).transpose(1, 2)
        key = self.key_linear(key).view(n_batches, key.size(1), self.h, d_k).transpose(1, 2)
        value = self.value_linear(value).view(n_batches, value.size(1), self.h, d_k).transpose(1, 2)

        # scores shape: (batch_size, h, max_sent_length, d_k)
        scores, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # concat attention scores over multiple heads
        # (batch_size, max_sent_length, d_model)
        scores = scores.transpose(1, 2).contiguous().view(n_batches, max_sent_length, self.h * d_k)

        return self.output_linear(scores)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size, max_len):
        super(Generator, self).__init__()
        self.linear = nn.Linear(in_features=d_model,
                                out_features=vocab_size)
        self.embed = torch.nn.Embedding(2,1)
        self.max_len=max_len
        self.pos_importance = nn.Linear(in_features=max_len,
                                out_features=max_len)
        self.relu = torch.nn.ReLU()
    def forward(self, x, src, exist_mask):

        #exist mask: BS,28-> BS,45,28
        # src: BS, 45-> BS, 45
        src = src!=0
        src = src.long()
        src = self.embed(src) #BS, 45, 1
        weight = self.relu(self.pos_importance(src.transpose(1,2))).transpose(1,2) #BS,45,1
        exist_mask = exist_mask.unsqueeze(1)
        exist_mask = exist_mask.repeat(1,self.max_len,1)

        x = self.linear(x)
        #x shape is BS,45,28
        #at each letter position, zero out the non candidates
        x = x.masked_fill_(exist_mask == 0, -1e9)
        x = F.log_softmax(x, dim=-1)
        x = torch.exp(x) #BS, 45, 28
        x = x.transpose(1,2) #BS, 28, 45

        #sum log probability at dim 1 with weights: BS, 28, 45 X 45, 1-> BS, 28, 1
        x = torch.matmul(x,weight)

        x = x.squeeze(-1)

        #output log_softmax logit
        x = F.log_softmax(x, dim=-1)

        # x = self.sum_para(torch.transpose(x,1,2))
        # x = x.squeeze(-1)

        #BS,28 -> BS,28
        # x = x.masked_fill_(exist_mask == 0, -1e9)
        # result = result.masked_fill_(exist_mask == 0, -1e9)
        # result, _ = torch.max(self.linear(x), dim=1)

        # result = result.masked_fill_(exist_mask == 0, -1e9)
        return F.log_softmax(x, dim=-1)
    

class SublayerSkipConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerSkipConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
class Encoder(nn.Module):
    def __init__(self, self_attn, feed_forward, size, dropout):
        super(Encoder, self).__init__()
        self.sub_layers = clone(SublayerSkipConnection(size, dropout), 2)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

    def forward(self, x, mask):
        x = self.sub_layers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sub_layers[1](x, self.feed_forward)
    

class FullyConnectedFeedForward(nn.Module):
    """
    A fully connected neural network with Relu activation
    input: d_model
    hidden: d_ff
    output: d_model

    Implements FFN equation.
    FFN(x) = max(0, xW_1 + b)W_2 + b

    It consist of two linear layer and a Relu activation in between

    Linear_2(Relu(Linear_1(x))))
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FullyConnectedFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x: shape (batch_size, max_sent_len, embedding_size/d_model)
        :return: output: shape (batch_size, max_sent_len, embedding_size/d_model)
        """
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    

    
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Bert(nn.Module):
    def __init__(self, encoder: nn.Module, generator, embedding, n_layers: int):
        """

        :param encoder: encoder/transformer layer that takes advantage of self-attention
        :param n_layers: int, number of encoder/transformer layers
        """
        super(Bert, self).__init__()
        self.encoder = encoder
        self.layers = clone(encoder, n_layers)
        self.embed = embedding
        self.layer_norm = LayerNorm(encoder.size)
        self.generator = generator

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor, candidates: torch.Tensor):
        """
        :param x: shape (batch_size, max_word_length)
        :param src_mask
        :return:
        """
        src = x
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, src_mask)

        x = self.layer_norm(x)
        x = self.generator(x=x,src=src,exist_mask=candidates)

        return x

    @property
    def device(self):
        return self.generator.linear.weight.device


def train_loop(data_loader, model, loss_fn, optimizer, loss_estimate, batch_no, epoch, epoch_no, device):
    size = len(data_loader.dataset)
    model.train()
    cnt = 0
    optimizer.zero_grad()
    loss = 0
    for batch in tqdm(data_loader):
        #mask is _, which is 1 in map
        X, candidates, y = batch
        src_mask = ((X != 1) & (X != 0)).unsqueeze(-2)
        pred = model(X,src_mask=src_mask, candidates=candidates)
        pred = torch.exp(pred) #BS,28
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        loss  = loss.item()+loss
        current = (cnt + 1) * len(X)
    
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(data_loader, model, loss_fn,t,device):
    size = 0
    model.load_state_dict(torch.load(f"model-{t}.pt"))
    model.eval()
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, candidates, y) in data_loader:
            #BS, 28
            src_mask = ((X != 1) & (X != 0)).unsqueeze(-2)
            pred = model(X,src_mask=src_mask, candidates=candidates)
            pred = torch.exp(pred)
            loss = loss_fn(pred,y)
            test_loss += loss.item()

            pred=torch.argmax(pred,-1)


            for i in range(len(pred)):
                guess = pred[i].item()
                if y[i][guess]>0:
                    correct +=1
            # y = y.masked_fill_(X!=1,-100)
            # correct += (pred.argmax(-1) == y).type(torch.float).sum().item()
            # size += (y != -100).type(torch.float).sum().item()
            size += len(pred)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomDatasetTrain(Dataset):
    def __init__(self, X_train, y_train):
        self.features = X_train
        self.label = y_train
        self.id = [i for i in range(28)]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.label[idx]
        # label_one_hot = [0 for i in range(len(self.id))]
        label_dist = [0 for i in range(len(self.id))]
        candidates = [1 for i in range(28)]
        for i in features:
            candidates[i]=0

        # for idx in label:
        #     if idx!=0 and idx !=1:
        #         label_one_hot[idx]=1

        for idx in label:
            if idx!=0 and idx !=1:
                label_dist[idx]=label_dist[idx]+1

        features = torch.tensor(features, dtype=torch.long).to(device)
        # label = torch.tensor(label, dtype=torch.long).to(device)
        # label_one_hot = torch.tensor(label_one_hot, dtype=torch.float).to(device)
        label_dist = torch.tensor(label_dist, dtype=torch.float).to(device)
        # list_of_candidates = []
        # for i in range(len(features)):
        #     if features[i]!=1:
        #         list_of_candidates.append([0 for j in range(len(self.id))])
        # list_of_candidates = torch.tensor(list_of_candidates, dtype=torch.long).to(device)
        candidates = torch.tensor(candidates, dtype=torch.long).to(device)
        return features, candidates, label_dist

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]


def create_dataloader(input_tensor, target_tensor, batch_size):
    all_features_data = CustomDatasetTrain(input_tensor, target_tensor)
    all_features_dataloader = DataLoader(all_features_data, batch_size=batch_size, shuffle=True)
    return all_features_dataloader

def save_model(model,t):
    torch.save(model.state_dict(), f"model-{t}.pt")

def train_model(train,test,model,device,batch_size,t):
    input_tensor, target_tensor = train
    all_features_dataloader_train = create_dataloader(input_tensor, target_tensor,batch_size)
    input_tensor, target_tensor = test
    all_features_dataloader_test = create_dataloader(input_tensor, target_tensor,batch_size)
    model = model.to(device)
    # criterion = nn.KLDivLoss(reduction='sum').to(device)
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    loss_estimate = []
    batch_no = []
    epoch_no = []
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(all_features_dataloader_train, model, loss_fn, optimizer, loss_estimate, batch_no, t, epoch_no,device)
    save_model(model,t)
    test_loop(all_features_dataloader_test, model, loss_fn,t,device)
    print("Done!")

def get_char_mapping():
    #create mapping e.g. _: 1, #: 0
    char2id = dict()
    char2id['_'] = 1
    char2id['#'] = 0
    char_list = string.ascii_lowercase
    for i, c in enumerate(char_list):
        char2id[c] = len(char2id)
    
    return char2id

def create_intermediate_data(df):
    x = pd.DataFrame(df)
    x[1] = x[0].apply(lambda p: len(p))
    x['vowels_present'] = x[0].apply(lambda p: set(p).intersection({'a', 'e', 'i', 'o', 'u'}))
    x['vowels_count'] = x['vowels_present'].apply(lambda p: len(p))
    x['unique_char_count'] = x[0].apply(lambda p: len(set(p)))
    x_ = x[~((x['unique_char_count'].isin([0, 1, 2])) | (x[1] <= 3)) & (x.vowels_count != 0)]

    return x

def loop_for_permutation(unique_letters, word, all_perm, i):
    random_letters = random.sample(unique_letters, i+1)
    new_permuted_word = word
    for letter in random_letters:
        new_permuted_word = new_permuted_word.replace(letter, "_")
        all_perm.append(new_permuted_word)

def permute_all(word, vowel_permutation_loop=False):
    unique_letters = list(set(word))
    all_perm = []
    if vowel_permutation_loop:
        for i in range(len(unique_letters) - 1):
            loop_for_permutation(unique_letters, word, all_perm, i)
        all_perm = list(set(all_perm))
        return all_perm
    else:
        for i in range(len(unique_letters) - 2):
            loop_for_permutation(unique_letters, word, all_perm, i)
        all_perm = list(set(all_perm))
        return all_perm

def permute_consonents(word):
    vowel_word = "".join([i if i in ["a", "e", "i", "o", "u"] else "_" for i in list(word)])
    vowel_idxs = []
    for i in range(len(vowel_word)):
        if vowel_word[i] == "_":
            continue
        else:
            vowel_idxs.append(i)
    abridged_vowel_word = vowel_word.replace("_", "")
    all_permute_consonents = permute_all(abridged_vowel_word, vowel_permutation_loop=True)
    permuted_consonents = []
    for permuted_word in all_permute_consonents:
        a = ["_"] * len(word)
        vowel_no = 0
        for vowel in permuted_word:
            a[vowel_idxs[vowel_no]] = vowel
            vowel_no += 1
        permuted_consonents.append("".join(a))
    return permuted_consonents

def create_masked_dictionary(df_aug):
    masked_dictionary = {}
    counter = 0
    for word in df_aug[0]:
        all_masked_words_for_word = []
        all_masked_words_for_word = all_masked_words_for_word + permute_all(word)
        all_masked_words_for_word = all_masked_words_for_word +  permute_consonents(word)
        all_masked_words_for_word = list(set(all_masked_words_for_word))
        masked_dictionary[word] = all_masked_words_for_word
        if counter % 10000 == 0:
            print(f"Mask: Iteration {counter} completed")
            #remove this break
            #break
        counter = counter + 1

    return masked_dictionary

def get_vowel_prob(df_vowel, vowel):
    try:
        return df_vowel[0].apply(lambda p: vowel in p).value_counts(normalize=True).loc[True]
    except:
        return 0

# def encode_output(word):
#     char_mapping = get_char_mapping()
#     output_vector = [0] * 26
#     for letter in word:
#         output_vector[char_mapping[letter]] = 1
#     return output_vector

def encode_output(word, max_len):
    char_mapping = get_char_mapping()
    while len(word)<max_len:
        word=word+"#"
    word_vector = []
    for letter_no in word:
        word_vector.append(char_mapping[letter_no])
    return word_vector

def encode_input(word, max_len):
    char_mapping = get_char_mapping()
    while len(word)<max_len:
        word=word+"#"
    word_vector = []
    for letter_no in word:
        word_vector.append(char_mapping[letter_no])
    return word_vector

def encode_words(masked_dictionary,max_len):
    target_data = []
    input_data = []
    for output_word, input_words in masked_dictionary.items():
        if len(input_words)>0:
            for input_word in input_words:
                if input_word.count("_")/len(input_word)<0.3:
                    input_data.append(encode_input(input_word,max_len))
                    output_vector = encode_output(output_word,max_len)
                    target_data.append(output_vector)

    print(f"Encode completed")
    return input_data, target_data


def convert_to_tensor(input_data, target_data, device):
    input_tensor = torch.tensor(input_data, dtype=torch.long).to(device)
    target_tensor = torch.tensor(target_data, dtype=torch.float32).to(device)
    return input_tensor, target_tensor

def get_datasets(df, device,max_len,vocab_size):
    x_ = create_intermediate_data(df)
    df_aug = x_.copy()
    masked_dictionary = create_masked_dictionary(df_aug)
    # vowel_prior = get_vowel_prior(df_aug)
    # save_vowel_prior(vowel_prior)

    return masked_dictionary
    # target_dist = []
    # for data in target_data:
    #     tgt = torch.tensor([data], dtype=torch.long, device=device)
    #     tgt_dist = convert_target_to_dist(tgt, vocab_size, mask=0, device=device)
    #     tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None])
    #     target_dist.append(tgt_dist)

    # target_dist=torch.stack(target_dist)

    # # save_input_output_data(input_data, target_data)
    # input_tensor, target_tensor = convert_to_tensor(input_data, target_data, device)

if __name__ == "__main__":
    f = open("word_list.txt", "r")
    df = []
    for x in f:
        df.append(x)
    for i in range(len(df)):
        df[i] = df[i].replace("\n", "")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device=torch.device('cpu')

    train, test= train_test_split(df, test_size=0.004)

    d_model = 256
    d_ff = 1024
    h = 4
    n_encoders = 4
    max_len = 45
    batch_size = 128
    vocab_size = 28
    epochs = 10

    train_dataset = get_datasets(train,device,max_len=max_len,vocab_size=vocab_size)
    test_dataset = get_datasets(test,torch.device('cpu'),max_len=max_len,vocab_size=vocab_size)



    self_attn = MultiHeadedAttention(h=h, d_model=d_model, d_k=d_model // h, d_v=d_model // h, dropout=0.1)
    feed_forward = FullyConnectedFeedForward(d_model=d_model, d_ff=d_ff)
    position = PositionalEncoding(d_model, dropout=0.1,max_len=max_len)
    embedding = nn.Sequential(Embeddings(d_model=d_model, vocab_size=vocab_size), position)

    encoder = Encoder(self_attn=self_attn, feed_forward=feed_forward, size=d_model, dropout=0.1)
    generator = Generator(d_model=d_model, vocab_size=vocab_size, max_len=max_len)
    model = Bert(encoder=encoder, embedding=embedding, generator=generator, n_layers=n_encoders)

    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    for t in range(epochs):
        train_dataset_encoded = encode_words(train_dataset,max_len)
        test_dataset_encoded = encode_words(test_dataset,max_len)
        train_model(train=train_dataset_encoded, test=test_dataset_encoded,model = model, device=device, batch_size=batch_size, t=t)