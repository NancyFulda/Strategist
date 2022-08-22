
#------------------------------------------------------------------------------------------
# Pre-trained model from https://github.com/huggingface/pytorch-pretrained-BERT#installation
#------------------------------------------------------------------------------------------


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from scipy import spatial
import pprint

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

def find_closest_point(vec, tokens, vectors):

    #this doesn't work because vectors are not normalized
    #distances = 1 - np.dot(vectors,vec.T)/np.linalg.norm(vec)

    #but this does
    distances = spatial.distance.cdist(vectors, [vec], metric='cosine')

    min_dist = np.nanmin(distances)
    index = np.where(distances == min_dist)[0][0]
    closest_word = tokens[index]
    closest_vector = vectors[index]

    return closest_vector

def compute_analogy(v, tokens, vectors):
    vec = v[1] - v[0] + v[2]

    filtered_tokens = tokens[:]
    filtered_vectors = vectors[:]

    for i in range(3):
        whr = np.where(abs(filtered_vectors - filtered_vectors[0]) < 0.00001)
        index = np.where(abs(filtered_vectors - v[i]) < 1e-5)[0][0]
        del filtered_tokens[index]
        del filtered_vectors[index]

    return all(find_closest_point(vec, filtered_tokens, filtered_vectors) == v[3])

def tokenize_text(text, tokenizer, naive_tokenization=False):
    # Tokenized input
    if naive_tokenization:
        tokenized_text = text.split(' ')
    else:
        tokenized_text = tokenizer.tokenize(text)
    #print(tokenized_text)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    #segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    sentences = ' '.join(tokenized_text).split('[SEP]')
    sentences = [s.strip().split() for s in sentences if s != '']
    segments_ids = []
    for i in range(len(sentences)):
        segments_ids += [i]*(len(sentences[i])+1)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensors

def get_hidden_states(model, tokens_tensor, segments_tensors):

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12

    return encoded_layers

def run_single_embedding(text, tokenizer, model):

    tokens_tensor, segments_tensors = tokenize_text(text, tokenizer)
    encoded_layers = get_hidden_states(model, tokens_tensor, segments_tensors)
    
    #quick hack to test encoding many sentence pairs at once
    #...but it didn't work
    #tt = torch.stack([tokens_tensor, tokens_tensor.detach().clone()])
    #st = torch.stack([segments_tensors, segments_tensors.detach().clone()])
    #encoded_layers = get_hidden_states(model, tt, st)

    return encoded_layers



if __name__ == '__main__':
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    run_single_embedding('this is a test [SEP]', tokenizer, model)
