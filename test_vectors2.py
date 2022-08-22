import numpy as np
from scipy import spatial
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import SIF_embedding
import pickle as pkl
import random
import pprint
import sys
import json

pp = pprint.PrettyPrinter()

#METHOD = 'USE_LARGE'
#METHOD = 'USE_LARGE_SIF'
#METHOD = 'USE_LARGE_MEAN'
#METHOD = 'USE_LITE'
#METHOD = 'BERT'
#METHOD = 'BERT_MEAN'
#METHOD = 'BERT_SIF'
#METHOD = 'BSE_bert'
#METHOD = 'BSE_encoder'
#METHOD = 'BSE_encoder_MEAN'
#METHOD = 'BSE_encoder_SIF'
#METHOD = 'FASTTEXT_BOW'
#METHOD = 'FASTTEXT_BOW_MEAN'
#METHOD = 'FASTTEXT_BOW_SIF'
#METHOD = 'GPT2_last'
#METHOD = 'GPT2_avg'
#METHOD = 'SPACY'
METHOD = 'SKIPTHOUGHT'

EMBEDDING_SIZE=300

print('/METHOD IS ' + METHOD)

CUDA=True
MAX_GOOGLE_SIZE=500

random.seed(13)

if METHOD == 'SKIPTHOUGHT':
    print("Importing Strategist...")
    import strategist
    s=strategist.Strategist(penseur=True,scholar=False)

if METHOD == 'SPACY':
    import spacy
    nlp = spacy.load('en')

if METHOD in ['BERT','BSE_bert','BERT_MEAN','BERT_SIF']:
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    from bert_embeddings_test import tokenize_text, get_hidden_states

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    if METHOD in ['BERT_SIF','BERT_MEAN']:
        filename='data/sample_sentences_augmented2.txt'
    
        X=[]
        with open(filename, 'r') as f:
            #f.readline()
            #for line in f:
            #    line = line.strip('\n')
            #    vector = np.array([float(x) for x in line.split()])
            #    X.append(vector)
            for line in f:
                sentence = line.strip('\n').strip()
                if len(sentence) > 1:
                    text = '[CLS] ' + sentence + ' [SEP]'
                    tokens_tensor, segments_tensors = tokenize_text(unicode(text), tokenizer)
                    encoded_layers = get_hidden_states(model, tokens_tensor, segments_tensors)
                    layer = encoded_layers[-1]
                    embedded_sentence = np.mean(np.squeeze(layer.cpu().numpy())[1:], axis=0)
                    X.append(embedded_sentence)
        X = np.vstack(X)
            
        if METHOD == 'BERT_SIF':
            import SIF_embedding
            x_SIF = SIF_embedding.compute_pc(X)

        if METHOD == 'BERT_MEAN':
            x_mean = np.mean(X, axis=0)

def fasttext_clean(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("'", " ' ")
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " ." )
    sentence = sentence.replace("!", " !" )
    sentence = sentence.replace("?", " ?" )
    sentence = sentence.replace(":", " :" )
    sentence = sentence.replace(";", " ;" )
    sentence = sentence.replace("-", " - " )
    sentence = sentence.replace("  ", " " )
    sentence = sentence.replace("  ", " " )
    return sentence.strip()

if METHOD in ['FASTTEXT_BOW','FASTTEXT_BOW_SIF','FASTTEXT_BOW_MEAN']:
    with open('data/fasttext.en.pkl','rb') as f:
        data = pkl.load(f)
        fasttext_tokens = data['tokens'][:50000]
        fasttext_vectors = data['vectors'][:50000]
    
    if METHOD in ['FASTTEXT_BOW_SIF','FASTTEXT_BOW_MEAN']:
        filename='data/sample_sentences_augmented2.txt'
    
        X=[]
        with open(filename, 'r') as f:
            for line in f:
                sentence = line.strip('\n').strip()
                if len(sentence) > 1:
                    vector = np.zeros(300)
                    sentence = fasttext_clean(sentence)
                    count = 0.0
                    for word in sentence.split():
                        try:
                            vector += fasttext_vectors[fasttext_tokens.index(word)]
                            count += 1
                        except:
                            pass

                    if count > 0:
                        X.append(vector/count)
                    else:
                        X.append(vector)
        X = np.vstack(X)
            
        if METHOD == 'FASTTEXT_BOW_SIF':
            import SIF_embedding
            x_SIF = SIF_embedding.compute_pc(X)

        if METHOD == 'FASTTEXT_BOW_MEAN':
            x_mean = np.mean(X, axis=0)

if METHOD in ['GPT2_last','GPT2_avg']:

    if sys.version_info[0] >= 3:
        raise Exception("The GPT encoder only works in Python 2")

    import torch
    from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model

    # Load pre-trained model (weights)
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if METHOD in ['BSE_encoder','BSE_encoder_SIF','BSE_encoder_MEAN']:
    import torch
    import dataset_fasttext

    if sys.version_info[0] < 3:
        raise Exception("The BSE_encoder only works in Python 3")

    # ** LOCAL **
    #model = 'output_autoencoder/_encoder.pkl'
    #model = 'output_autoencoder_plus_w2v/_encoder.pkl'
    #model = 'output_junk/2_encoder.pkl'
    #model = 'output_context/output_context1_encoder.pkl'
    #model = 'output_w2v/output_w2v1_encoder.pkl'
    #bse_encoder = torch.load(model)
    
    # ** Fifth Round **
    #model = 'output_autoencoder_book_corpus_baseline/3_encoder.pkl'
    #model = 'output_autoencoder_book_corpus_no_fasttext/4_encoder.pkl'
    #model = 'output_autoencoder_book_corpus_z_mu/10_encoder.pkl'
    #model = 'output_autoencoder_wikipedia/4_encoder.pkl'
    #model = 'output_autoencoder_wikipedia_no_fasttext/4_encoder.pkl'
    #model = 'output_w2v_book_corpus/4_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus/4_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_low_learning_rate/3_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_no_fasttext/7_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_z_mu/5_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_wikipedia/3_encoder.pkl'
    #model = 'output_w2v_wikipedia/3_encoder.pkl'
    #model = 'output_w2v_wikipedia_no_fasttext/5_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_high_sample_freq/4_encoder.pkl'

    #bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v5/' + model)

    # ** Sixth Round **
    #model = 'output_w2v_book_corpus_z_mu/200_encoder.pkl'
    #model = 'output_w2v_wikipedia_z_mu/156_encoder.pkl'
    #model = 'output_w2v_book_corpus_z_mu_higher_learning_rate/57_encoder.pkl'
    #model = 'output_w2v_plus_neg_wikipedia_z_mu/9_encoder.pkl'
    #model = 'output_autoencoder_plus_neg_wikipedia_z_mu/20_encoder.pkl'
    #model = 'output_autoencoder_plus_neg_wikipedia_z_mu_trial2/16_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_plus_neg_wikipedia_z_mu_trial2/4_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_split_corpus_1_z_mu/23_encoder.pkl'

    #bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v6/' + model)

    # ** Seventh Round **
    model = 'output_autoencoder_book_corpus/7_encoder.pkl'
    #model = 'output_context_plus_inv_text_book_corpus/16_encoder.pkl'
    #model = 'output_context_plus_inv_text_book_corpus_trial2/16_encoder.pkl'
    #model = 'output_context_plus_w2v_book_corpus/27_encoder.pkl'
    #model = 'output_context_plus_w2v_plus_inv_text_book_corpus/18_encoder.pkl'
    #model = 'output_context_plus_w2v_plus_inv_text_book_corpus_higher_context_loss/16_encoder.pkl'
    ##model = 'output_context_plus_w2v_split_corpus_1/0_encoder.pkl'
    ##model = 'output_inv_text_plus_autoencoder_book_corpus/0_encoder.pkl'
    ##model = 'output_inv_text_plus_w2v_book_corpus/0_encoder.pkl'
    #model = 'output_w2v_book_corpus/219_encoder.pkl'
    #model = 'output_w2v_plus_context_book_corpus/15_encoder.pkl'
    
    bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v7/' + model)

    # ** Eighth Round **
    #model = 'output_context_book_corpus/5_encoder.pkl'
    #model = 'output_context_book_corpus_bigger_learning_rate/7_encoder.pkl'
    ##model = 'output_context_plus_w2v_split_corpus_1/0_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_1_w2v_overdrive_10/52_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_1_w2v_overdrive_100/133_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_2/18_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_2_w2v_overdrive_10/141_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_2_w2v_overdrive_100/114_encoder.pkl'
    #model = 'output_inv_text_plus_autoencoder_book_corpus/9_encoder.pkl'
    #model = 'output_inv_text_plus_context_book_corpus/9_encoder.pkl'
    ##model = 'output_inv_words_plus_autoencoder_book_corpus/0_encoder.pkl'
    ##model = 'output_inv_words_plus_context_book_corpus/0_encoder.pkl'

    #bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v8/' + model)


    print('/model is ' + model)
    bse_encoder.eval()
    bse_encoder.cuda()
    
    print('Done loading model')

    dataset=dataset_fasttext.Dataset()

    if METHOD in ['BSE_encoder_SIF','BSE_encoder_MEAN']:
        filename='data/sample_sentences_augmented2.txt'
        X = []
        f = open(filename,'r')
        for line in f:
            sentence = line.strip('\n').strip()
            if len(sentence) > 1:
                x = bse_encoder.ftext.get_indices(sentence)
                x = dataset.to_onehot(x)
                vector = bse_encoder([x.cuda()]).data.cpu().numpy() if CUDA else bse_encoder([x]).data.numpy()
                #vector = embedding.data.view(600)
                #print(vector.shape)
                X.append(vector)
            
        if METHOD == 'BSE_encoder_SIF':
            import SIF_embedding
            x_SIF = SIF_embedding.compute_pc(np.vstack(X))
        if METHOD == 'BSE_encoder_MEAN':
            x_MEAN = np.mean(np.vstack(X),axis=0)

if METHOD in ['USE_LARGE','USE_LARGE_SIF','USE_LARGE_MEAN']:
    import tensorflow as tf
    import tensorflow_hub as hub
    tf_embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    
    if METHOD in ['USE_LARGE_SIF','USE_LARGE_MEAN']:
        filename='data/sample_sentences_augmented2.txt'
        sample_sentences = []
        f = open(filename,'r')
        for line in f:
            sentence = line.strip('\n').strip()
            if len(sentence) > 1:
                sample_sentences.append(sentence)

        with tf.Session() as session:
            X = tf_embed(sample_sentences)
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            session.run(X)
        
            if METHOD == 'USE_LARGE_SIF':
                import SIF_embedding
                x_SIF = SIF_embedding.compute_pc(X.eval())
            if METHOD == 'USE_LARGE_MEAN':
                x_mean = np.mean(X.eval(), axis=0)

if METHOD == 'USE_LITE':
    import embedding_apis as embed_api

if METHOD == 'BSE_bert':
    #infile = 'checkpoints/antonym_test/W.pkl'
    infile = 'checkpoints/negations_1024/W.pkl'
    f=open(infile,'r')
    W=pkl.load(f)


#METRIC ='euclidean'
METRIC ='cosine'

def skipthought_embed(text):
    return s.encode(text)

def spacy_embed(text):
    return nlp(unicode(''.join([c for c in text if ord(c)<128]))).vector

def bert_embed(text):
    text = '[CLS] ' + text + ' [SEP]'

    #strip non-unicode characters
    text = ''.join(i for i in text if ord(i)<128)

    tokens_tensor, segments_tensors = tokenize_text(unicode(text), tokenizer)
    encoded_layers = get_hidden_states(model, tokens_tensor, segments_tensors)

    layer = encoded_layers[-1]
    embedded_sentence = np.mean(np.squeeze(layer.cpu().numpy())[1:], axis=0)
    
    if METHOD == 'BERT_MEAN':
        return embedded_sentence - x_mean
    elif METHOD == 'BERT_SIF':
        return SIF_embedding.remove_pc(np.array(embedded_sentence).reshape(1,-1), npc=1, pc=x_SIF)
    else:
        return embedded_sentence

def gpt2_embed(text):

    # Encode some inputs
    #text_1 = "Who was Jim Henson ?"
    #text_2 = "Jim Henson was a puppeteer"
    indexed_tokens = tokenizer.encode(text)
    #indexed_tokens_1 = tokenizer.encode(text_1)
    #indexed_tokens_2 = tokenizer.encode(text_2)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    #tokens_tensor_1 = torch.tensor([indexed_tokens_1])
    #tokens_tensor_2 = torch.tensor([indexed_tokens_2])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    #tokens_tensor_1 = tokens_tensor_1.to('cuda')
    #tokens_tensor_2 = tokens_tensor_2.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        hidden_states, past = model(tokens_tensor)
        #hidden_states_1, past = model(tokens_tensor_1)
        # past can be used to reuse precomputed hidden state in a subsequent predictions
        # (see beam-search examples in the run_gpt2.py example).
        #hidden_states_2, past = model(tokens_tensor_2, past=past)

    hidden_states=hidden_states.reshape(-1,768)
    
    #print(text)
    #print(len(text.split()))
    #print('Aquired hidden states!')
    #print(hidden_states.shape)

    if METHOD == 'GPT2_last':
        return hidden_states[-1].cpu().numpy()
    elif METHOD == 'GPT2_avg':
        hidden_states=np.mean(hidden_states.cpu().numpy(),axis=0)
        return hidden_states
    else:
        raise ValueError('Unknown METHOD ' + METHOD)


def bse_encoder_embed(text):
    bse_encoder.encoder_rnn.rnn.flatten_parameters()
    
    x = bse_encoder.ftext.get_indices(text)
    #x = bse_encoder.dataset.to_onehot(x)
    x = dataset.to_onehot(x)
    if CUDA:
        embedding = bse_encoder([x.cuda()]).data.cpu().numpy()
    else:
        embedding = bse_encoder([x]).data.numpy()
    if METHOD == 'BSE_encoder':
        return embedding.reshape(EMBEDDING_SIZE)
    elif METHOD == 'BSE_encoder_SIF':
        new_embedding = SIF_embedding.remove_pc(np.array(embedding).reshape([1,-1]), npc=1, pc=x_SIF)
        return new_embedding.reshape(EMBEDDING_SIZE)
    elif METHOD == 'BSE_encoder_MEAN':
        new_embedding = embedding = x_MEAN
        return new_embedding.reshape(EMBEDDING_SIZE)

def bse_bert_embed(text):
    bert_embedding = bert_embed(text)
    embedded_sentence = np.sum(np.multiply(bert_embedding, W.T), axis=1)
    return embedded_sentence

def fasttext_embed(text):
    vector = np.zeros(300)
    text = fasttext_clean(text)
    count = 1
    for word in text.split():
        try:
            vector += fasttext_vectors[fasttext_tokens.index(word)]
            count += 1.0
        except:
            pass

    if count > 0:
        vector = vector/count
    if np.sum(vector) == 0:
        vector = vector+1e5

    if METHOD == 'FASTTEXT_BOW':
        return vector
    elif METHOD == 'FASTTEXT_BOW_SIF':
        new_vector = SIF_embedding.remove_pc(np.array(vector).reshape([1,-1]), npc=1, pc=x_SIF)
        return new_vector.reshape(300)
    elif METHOD == 'FASTTEXT_BOW_MEAN':
        new_vector = vector = x_mean
        return new_vector.reshape(300)

def use_large_embed(text):
    with tf.Session() as session:
        if isinstance(text, basestring):
            embeddings = tf_embed([text])
        else:
            # we assume it's a list of strings
            embeddings = tf_embed(text)
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        if METHOD == 'USE_LARGE':
            return session.run(embeddings)
        elif METHOD == 'USE_LARGE_SIF':
            print(x_SIF.shape)
            print(embeddings.eval().shape)
            return SIF_embedding.remove_pc(embeddings.eval(), npc=1, pc=x_SIF)
        elif METHOD == 'USE_LARGE_MEAN':
            return embeddings.eval()-x_mean

def use_lite_embed(text):
    return embed_api.embed([text], 'use_lite')[0]

def embed(text):
    if METHOD== 'BSE_bert':
        return bse_bert_embed(text)
    elif METHOD in ['BERT','BERT_MEAN','BERT_SIF']:
        return bert_embed(text)
    elif METHOD in ['USE_LARGE','USE_LARGE_SIF','USE_LARGE_MEAN']:
        return use_large_embed(text)
    elif METHOD in ['USE_LITE']:
        return use_lite_embed(text)
    elif METHOD in ['BSE_encoder','BSE_encoder_SIF','BSE_encoder_MEAN']:
        return bse_encoder_embed(text)
    elif METHOD in ['FASTTEXT_BOW','FASTTEXT_BOW_SIF','FASTTEXT_BOW_MEAN']:
        return fasttext_embed(text)
    elif METHOD in ['GPT2_last','GPT2_avg']:
        return gpt2_embed(text)
    elif METHOD == 'SPACY':
        return spacy_embed(text)
    elif METHOD == 'SKIPTHOUGHT':
        return skipthought_embed(text)
    else:
        raise ValueError('No valid embedding function found for METHOD ' + METHOD)
    # return spacy_embed(text)
    # return word2vec_embed(text)
    # return use_lite_embed(text)
    # return elmo_embed(text)
    # return GPT_embed(text)
    # return GPT2_embed(text)

def list_embed(text_list):
    if METHOD in ['USE_LARGE','USE_LARGE_SIF','USE_LARGE_MEAN']:
        return embed(text_list)
    else: 
        embeddings=[]
        for t in text_list:
            embeddings.append(embed(t))
        return embeddings

def test_sentence_pair(s1,s2,output='all'):
    v1 = embed(s1)
    v2 = embed(s2)
    if METRIC == 'cosine':
        dist = spatial.distance.cosine(v1,v2)
    elif METRIC == 'euclidean':
        dist = spatial.distance.euclidean(v1,v2)
    else:
        raise ValueError('Unknown metric: ' + METRIC)

    print_output = False
    if output == 'all':
        print_output = True
    if output == 'random':
        if random.randint(0,10)>8:
            print_output = True
    if print_output:
        print(s1)
        print(s2)
        print(dist)
        print('\n')
    return dist

def find_nearest_match(input_, targets, embedded_targets, num_matches):
    #if isinstance(input_, basestring):
    if isinstance(input_, str):
        embedding = embed(input_)
    else:
        embedding = input_

    if METHOD in ['USE_LARGE','BERT_SIF','USE_LARGE_SIF','USE_LARGE_MEAN','FASTTEXT_BOW','BSE_encoder','BSE_encoder_SIF','BSE_encoder_MEAN','GPT2']:
        new_list = []
        for e in embedded_targets:
            new_list.append(np.squeeze(e))
        embedded_targets = new_list

    #print('here')
    #print(embedding.shape)

    distances = spatial.distance.cdist(embedded_targets, [np.squeeze(embedding)], metric=METRIC)

    closest_matches = []
    distance_list = []
    for i in range(num_matches):
        min_dist = np.nanmin(distances)
        index = np.where(distances == min_dist)[0][0]
        closest_match = targets[index]
        closest_vector = embedded_targets[index]

        distances[index] = 1000
        closest_matches.append(closest_match)
        distance_list.append(min_dist)

    return closest_matches, distance_list
   

def run_quadcopter_test(n):
    targets = []
    targets.append('go forward')
    targets.append('go backward')
    targets.append('move faster')
    targets.append('move slower')
    targets.append('go left')
    targets.append('go right')
    targets.append('go higher')
    targets.append('go lower')
    targets.append('stop')

    sentences = []
    sentences.append(('go forward ten feet', 'go forward'))
    sentences.append(('move a bit forward', 'go forward'))
    sentences.append(('now go forward', 'go forward'))
    sentences.append(('can you go a bit farther?', 'go forward'))
    sentences.append(('can you go a bit farther forward?', 'go forward'))
    sentences.append(('foward, please', 'go forward'))
    sentences.append(('advance three feet', 'go forward'))
    sentences.append(('scoot forward a bit', 'go forward'))
    sentences.append(('back up', 'go backward'))
    sentences.append(('no, that\'s wrong, go back', 'go backward'))
    sentences.append(('reverse', 'go backward'))
    sentences.append(('backward', 'go backward'))
    sentences.append(('back', 'go backward'))
    sentences.append(('go the other way', 'go backward'))
    sentences.append(('turn around', 'go backward'))
    sentences.append(('can you back up a few meters?', 'go backward'))
    sentences.append(('speed up', 'go faster'))
    sentences.append(('that\'s too slow', 'go faster'))
    sentences.append(('that\'s too slow, pick up the pace', 'go faster'))
    sentences.append(('pick up the pace', 'go faster'))
    sentences.append(('increase velocity', 'go faster'))
    sentences.append(('get your rear in gear, robot!', 'go faster'))
    sentences.append(('slow down!', 'go slower'))
    sentences.append(('slower, please', 'go slower'))
    sentences.append(('that\'s too fast', 'go slower'))
    sentences.append(('can you slow down a bit?', 'go slower'))
    sentences.append(('reduce velocity', 'go slower'))
    sentences.append(('woah, slow down', 'go slower'))
    sentences.append(('turn right', 'go right'))
    sentences.append(('now go five meters to the right', 'go right'))
    sentences.append(('turn to your right', 'go right'))
    sentences.append(('right-hand turn', 'go right'))
    sentences.append(('make a sharp turn to you right', 'go right'))
    sentences.append(('turn left', 'go left'))
    sentences.append(('now go five meters to the left', 'go left'))
    sentences.append(('turn to your left', 'go left'))
    sentences.append(('left-hand turn', 'go left'))
    sentences.append(('make a sharp turn to your left', 'go left'))
    sentences.append(('can you go a bit higher?', 'go higher'))
    sentences.append(('higher, please', 'go higher'))
    sentences.append(('try to touch the ceiling', 'go higher'))
    sentences.append(('increase altitude', 'go higher'))
    sentences.append(('you\'re too low', 'go higher'))
    sentences.append(('can you get a bit further up?', 'go higher'))
    sentences.append(('lower, please', 'go lower'))
    sentences.append(('land on the ground', 'go lower'))
    sentences.append(('decrease altitude', 'go lower'))
    sentences.append(('that\'s too high', 'go lower'))
    sentences.append(('can you get a bit farther down?', 'go lower'))
    sentences.append(('drop down a few feet', 'go lower'))
    sentences.append(('all right, stop', 'stop'))
    sentences.append(('maintain position', 'stop'))
    sentences.append(('don\'t move', 'stop'))
    sentences.append(('ok, that\'s good, don\'t move', 'stop'))
    sentences.append(('Woah!', 'stop'))
    sentences.append(('stay right there.', 'stop'))
    sentences.append(('wait there.', 'stop'))
    sentences.append(('hold still', 'stop'))

    embedded_targets = []
    for t in targets:
        embedded_targets.append(embed(t))

    num_correct=0
    for sentence, answer in sentences:
        matches, distances = find_nearest_match(sentence, targets, embedded_targets, num_matches=n)
        print('\n' + sentence)
        for m, d in zip(matches, distances):
            print('\t%4f ==> %s' % (d, m))
        if matches[0] == answer:
            num_correct += 1

        # RANDOM BASELINE
        #if random.choice(targets) == answer:
        #    num_correct += 1

    print('%d / %d correct' %(num_correct, len(sentences)))

def sentence_analogy(src_1, src_2, tgt_1, tgt_2, sample_sentences, sample_vectors, n=10, output='all'):
    if isinstance(src_1, str):
        v_src_1 = embed(src_1)
        v_src_2 = embed(src_2)
        v_tgt_1 = embed(tgt_1)
    else:
        # assume we were given vectors
        v_src_1 = src_1
        v_src_2 = src_2
        v_tgt_1 = tgt_1
    v_tgt_2 = v_src_2 - v_src_1 + v_tgt_1
    matches, distances = find_nearest_match(v_tgt_2, sample_sentences, sample_vectors, num_matches=n)

    shall_we_print=False
    if output=='all':
        shall_we_print = True
    elif output == 'random':
        if random.randint(0,1000) > 995:
            shall_we_print = True

    if shall_we_print:
        print('\n')
        print(src_1, src_2, tgt_1)
        for m, d in zip(matches, distances):
            print('\t'+str(d)+'\t'+m)

    #calculate scores
    if tgt_2 in matches[:3]:
        return 1
    else:
        return 0

    #RANDOM BASELINE
    """if tgt_2 in random.sample(sample_sentences, 3):
        return 1
    else:
        return 0"""

def get_sample_set(filename):
    sample_sentences = []
    sample_vectors = []
    f = open(filename,'r')
    for line in f:
        sentence = line.strip('\n').strip()
        if len(sentence) > 1:
            sample_sentences.append(sentence)
    if METHOD in ['USE_LARGE','USE_LARGE_SIF','USE_LARGE_MEAN']:
        sample_vectors = embed(sample_sentences)
    else:
        sample_vectors = []
        for s in sample_sentences:
            vector = embed(s)
            sample_vectors.append(vector)

    #HACK to test mini-google
    #sample_sentences = []

    #Now add in the google analogy words
    google_sentences = []
    f = open('data/google_analogy_test_set.txt','r')
    categories = f.read().split(':')[1:]
    for cat in categories:
        data = cat.split('\n')[:MAX_GOOGLE_SIZE]
        for analogy in data[1:]:
            words = analogy.split()
            for w in words:
                if len(w)>0 and w.lower() not in google_sentences:
                    google_sentences.append(w.lower())

    google_sentences = list(set(google_sentences))
    if METHOD not in ['USE_LARGE','USE_LARGE_SIF','USE_LARGE_MEAN']:
        google_vectors = []
        for s in google_sentences:
            vector = embed(s)
            google_vectors.append(vector)

    if METHOD in ['USE_LARGE','USE_LARGE_SIF','USE_LARGE_MEAN']:
        google_vectors = embed(google_sentences)

    print(str(len(sample_sentences)) + ' sample sentences found.')
    print(str(len(google_sentences)) + ' google words found.')
    return sample_sentences, sample_vectors, google_sentences, google_vectors

def test_similarity_metrics():
    score=0
    count=0

    dist1 = test_sentence_pair('the cat chased the dog','the dog chased the cat')
    dist2 = test_sentence_pair('In Tahiti, the cat chased the dog','The cat chased the dog in Tahiti')
    count += 1
    if dist1 > dist2:
        score += 1
    
    #dist1 = test_sentence_pair('I bought two apples at the store','I bought two pears at the store yesterday')
    #dist2 = test_sentence_pair('I bought two apples at the store','At the store, I bought two apples')
    #count += 1
    #if dist1 > dist2:
    #    score += 1

    dist1 = test_sentence_pair('I am a cat','I am not a cat')
    dist2 = test_sentence_pair('I am a cat','I am a domesticated cat')
    count += 1
    if dist1 > dist2:
        score += 1

    dist1 = test_sentence_pair('i am happy','i am sad')
    dist2 = test_sentence_pair('i am happy','i am not sad')
    count += 1
    if dist1 > dist2:
        score += 1

    dist1 = test_sentence_pair('king', 'beggar')
    dist2 = test_sentence_pair('prince', 'merchant')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('rich', 'poor')
    dist2 = test_sentence_pair('wealthy', 'middle class')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('wealthy', 'impoverished')
    dist2 = test_sentence_pair('wealthy', 'middle class')
    count += 1
    if dist1 > dist2:
        score += 1

    dist1 = test_sentence_pair('left', 'right')
    dist2 = test_sentence_pair('left', 'middle')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('yes', 'no')
    dist2 = test_sentence_pair('yes', 'maybe')
    count += 1
    if dist1 > dist2:
        score += 1

    dist1 = test_sentence_pair('of course', 'absolutely not')
    dist2 = test_sentence_pair('of course', 'absolutely')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('of course not', 'absolutely')
    dist2 = test_sentence_pair('of course', 'absolutely')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('i like him', 'i don\'t like him')
    dist2 = test_sentence_pair('i like him', 'i love him')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('i like him', 'i do not like him')
    dist2 = test_sentence_pair('i like him', 'i love him')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('i like him', 'i hate him')
    dist2 = test_sentence_pair('i like him', 'i dislike him')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('hot', 'cold')
    dist2 = test_sentence_pair('hot', 'warm')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('the milk is hot', 'the milk is cold')
    dist2 = test_sentence_pair('the milk is hot', 'the milk is warm')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('my friends don\'t like you', 'the milk is warm')
    dist2 = test_sentence_pair('my friends don\'t like you', 'my enemies don\'t like you')
    count += 1
    if dist1 > dist2:
        score += 1

    dist1 = test_sentence_pair('i own a dog', 'i own a bird')
    dist2 = test_sentence_pair('i own a dog', 'i own a cocker spaniel')
    count += 1
    if dist1 > dist2:
        score += 1
    
    dist1 = test_sentence_pair('i own a dog', 'i own a bird')
    dist2 = test_sentence_pair('i own a dog', 'i own a dachshund')
    count += 1
    if dist1 > dist2:
        score += 1
    

    #print('\n******************\n')
    #test_sentence_pair('I am a cat','I am a dog')
    #test_sentence_pair('i am not happy','i am not sad')
    #test_sentence_pair('The religious man built a castle', 'the irreligious man destroyed a castle')
    #test_sentence_pair('I felt hot, so I went into the water', 'I felt cold, so I went out of the water')
    #test_sentence_pair('happy', 'sad')
    #test_sentence_pair('up', 'down')
    #test_sentence_pair('king', 'queen')
    print('\ndistance tests;')
    print(str(score) + '/' + str(count))

def test_commonsense_reasoning():
    score = 0
    score += sentence_analogy('to touch the ceiling', 'you must go higher','to touch the floor','you must go lower', ss, sv)
    score += sentence_analogy('if you drop a ball', 'it will bounce','if you drop a mirror','it will shatter', ss, sv)
    score += sentence_analogy('if i help you', 'we will be friends','if i hit you','we will be enemies', ss, sv)
    score += sentence_analogy('a ladder', 'can be climbed','a river','can be crossed', ss, sv)
    score += sentence_analogy('that book was great', 'the author is very talented','that song was fantastic','the lead singer is very talented',ss,sv)
    score += sentence_analogy('i am happy', 'i am not happy','i am sad','i am not sad', ss, sv)
    score += sentence_analogy('i am happy', 'i am sad','i am not happy','i am not sad', ss, sv)
    score += sentence_analogy('heal', 'hurt','fix','break', ss, sv)
    score += sentence_analogy('if you water a plant', 'it will grow','if you cut me','i will bleed', ss, sv)
    score += sentence_analogy('water plant', 'it will grow','cut me','i will bleed', ss, sv)
    score += sentence_analogy('man', 'king','woman','queen', ss, sv)
    score += sentence_analogy('brush', 'paint','pencil','draw', ss, sv)
    score += sentence_analogy('ceiling', 'up','floor','down', ss, sv)
    score += sentence_analogy('a king', 'is wealthy','a peasant','is poor', ss, sv)
    score += sentence_analogy('enter', 'door','climb','ladder', ss, sv)
    score += sentence_analogy('to enter a room', 'you use the door','to climb a roof','you use a ladder', ss, sv)
    score += sentence_analogy('try to kill all the monsters', 'kill monster','try to get all the scrolls','get scroll', ss, sv)
    print('\Commonsense reasoning')
    print (str(score) + '/17')


def SAT_analogy(source, targets, answer):
    analogy_vector1 = embed(source[1]) - embed(source[0])
    min_dist = 1000
    idx = None
    distances = []
    for i, t in enumerate(targets):
        try:
            #analogy_vector2 = embed(t[0])-embed(source[0])
            test_vector1 = embed(t[1]) - embed(t[0])
            #test_vector2 = embed(t[1]) - embed(source[1])
        except:
            print(source)
            print(answer)
            print(targets)
            print(t)
            sys.exit()
        dist = spatial.distance.cosine(test_vector1,analogy_vector1)
        #dist = spatial.distance.cosine(test_vector1,analogy_vector1) + spatial.distance.cosine(test_vector2,analogy_vector2)
        distances.append(dist)
        if dist < min_dist:
            min_dist = dist
            idx = i

    #print(source)
    #print(targets)
    #print(distances)
    #print(answer)
    #print(idx)
    #raw_input('>')

    if idx is not None:
        return idx == answer
    else:
        return 0

def test_SAT():
    from itertools import groupby
    with open('data/SAT-package-V3.txt','r') as f:
        data = f.readlines()

    #cut out the intro text
    data = data[41:]
    data = [d.strip('\n').strip('\r') for d in data]
    questions = [list(group) for k, group in groupby(data, lambda x: x== '') if not k]

    score = 0
    count = 0
    for question in questions:
        count += 1
        source = question[1].split()[:2]
        targets = [q.split()[:2] for q in question[2:]]
        answer = ord(targets.pop()[0])-97 #convert letters to indexes
        #try:
        score += SAT_analogy(source, targets, answer)
        #except:
        #    pass

    #print('SAT score: ' + str(score) + ' / ' + str(count))
    print('/SAT score: '+str(100*float(score)/float(count))+'%')

def test_google(sample_sentences, sample_vectors, n=-1):
    f = open('data/google_analogy_test_set.txt','r')
    categories = f.read().split(':')
    results_dict = {}
    total_correct = 0
    total_count = 0
    total_omitted = 0
    for cat in categories:
        data = cat.split('\n')[:MAX_GOOGLE_SIZE]
        category_name = data[0]
        if len(data)<10:
            continue
        if n>0:
            data = random.sample(data[1:], min(len(data)-2,n))
        else:
            data = data[1:]

        count = len(data)
        num_correct = 0
        num_omitted = 0

        analogy_words = []
        for analogy in data[1:]:
            words = analogy.split()
            if len(words) != 4:
                continue

            words = [w.lower() for w in words]
            analogy_words.append(words)

        analogy_words = np.vstack(analogy_words)
        orig_shape = analogy_words.shape
        analogy_vectors=np.vstack(list_embed(analogy_words.reshape(-1)))

        for analogy,vectors in zip(analogy_words,analogy_vectors.reshape([orig_shape[0],orig_shape[1],analogy_vectors[0].shape[-1]])):
            num_correct += sentence_analogy(vectors[0], vectors[1], vectors[2], analogy[3], sample_sentences, sample_vectors, output='random')
        #for analogy in analogy_words:
        #    vectors = list_embed(analogy)
        #    num_correct += sentence_analogy(vectors[0], vectors[1], vectors[2], analogy[3], sample_sentences, sample_vectors, output='random')

            #RANDOM BASELINE
            #num_correct += (random.choice(sample_sentences) == words[3])

        total_correct += num_correct
        total_count += count
        total_omitted += num_omitted
        #print('num correct %d/%d, %f' % (num_correct, count, float(num_correct)/count))

        results_dict[data[0]] = str(num_correct) + ' correct, ' + str(count) + ' total, ' + str(100 * float(num_correct)/count) + '%, omitted ' + str(num_omitted)

    print('\n\nmini-google')
    pp.pprint(results_dict)
    print('/Full Google Composite %f%s' % (100 * float(total_correct)/total_count, '%'))
    print('/Google Omissions %d' % (total_omitted))

def test_entailment():
    with open('evaluation_sets/snli_1.0/snli_1.0_test.jsonl','r') as f:
        data = f.read()
        lines = data.split('\n')
        json_lines = [json.loads(line) for line in lines[:-1]]

    X=[]
    Y=[]
    e_Y=[]
    vals={'entailment':0.0,'neutral':0.5,'contradiction':1.0}
    for j in json_lines[:]:
        s1 = embed(j['sentence1'])
        s2 = embed(j['sentence2'])
        dist = spatial.distance.cosine(s1,s2)
        e_vector = s2-s1
        #e_vec_dist = spatial.distance.cosine(entailment_vector, e_vector)
        #print('/z,n=', '%.4f\t%.4f\%%.4f'%(z_score,n_dist,e_vec_dist), j['gold_label'])
        if j['gold_label'] != '-':
            X.append(vals[j['gold_label']])
            Y.append(dist)
            #e_Y.append(e_vec_dist)
    
    corr, p_value=pearsonr(X,Y)
    #e_corr, p_value=pearsonr(X,e_Y)
    #print('\n/entailment n = ' + str(n))
    print('/entailment r: ',corr)
    #print('/entailment pearson\'s r: ',e_corr)

    rho, p_value=spearmanr(X,Y)
    #e_rho, p_value=spearmanr(X,Y)
    print('/entailment rho: ',rho)
    #print('/entailment spearman\'s rho: ',e_rho)
    print('/')

def test_wordsim353():
    f = open('evaluation_sets/wordsim353/set2.csv','r')
    data = f.read()
    lines = data.split('\n')[1:-1]
    
    X = []
    Y = []
    for l in lines:
        items = l.split(',')
        w1 = embed(items[0].lower())
        w2 = embed(items[1].lower())
        dist = spatial.distance.cosine(w1,w2)
        X.append(dist)
        Y.append(float(-1.0*float(items[2])))
        #print('%.4f'%(dist), items[2])

    corr, p_value=pearsonr(X,Y)
    print('/wordsim r: ',corr)

    rho, p_value=spearmanr(X,Y)
    print('/wordsim rho: ',rho)
    print('/')

def test_simlex():
    f = open('evaluation_sets/SimLex-999/SimLex-999.txt','r')
    data = f.read()
    lines = data.split('\n')[1:-1]
    
    X = []
    Y = []
    count = 0
    omissions=0
    for l in lines[:-1]:
        count += 1
        items = l.split('\t')
        w1 = embed(items[0].lower())
        w2 = embed(items[1].lower())
        dist = spatial.distance.cosine(w1,w2)
        """if np.isnan(dist):
            print('/',dist)
            print('/', items[0], len(items[0]))
            print('/', items[1], len(items[1]))
            print('/',w1)
            print('/',w2)
            print('/',items)
            input('>')"""
        X.append(dist)
        Y.append(-1.0*float(items[3]))
        omissions += 1
        #print('%.4f'%(dist), items[2])

    #sx = np.std(np.array(X))
    #sy = np.std(np.array(Y))
    #cov = np.mean((np.array(X)-np.mean(np.array(X)))*(np.array(Y)-np.mean(np.array(Y))))
    #corr = cov/(sx*sy)
    #print('/', sx, sy, cov, corr)

    corr, p_value=pearsonr(X,Y)
    print('/SimLex r: ',corr)

    rho, p_value=spearmanr(X,Y)
    print('/SimLex rho: ',rho)
    print('/SimLex omissions: ', omissions, 'out of ', count)
    print('/')

def test_sts_benchmark():
    f = open('evaluation_sets/stsbenchmark/sts-test.csv','r')
    data = f.read()
    lines = data.split('\n')[1:-1]
    
    X = []
    Y = []
    for l in lines:
        items = l.split('\t')
        w1 = embed(items[5].lower())
        w2 = embed(items[6].lower())
        dist = spatial.distance.cosine(w1,w2)
        X.append(dist)
        Y.append(float(-1.0*float(items[4])))
        #print('%.4f'%(dist), items[2])

    corr, p_value=pearsonr(X,Y)
    print('/STS-benchmark r: ',corr)

    rho, p_value=spearmanr(X,Y)
    print('/STS-benchmark rho: ',rho)
    print('/')

def test_ant_syn_sentences():
    f = open('evaluation_sets/ant_syn_sentences.txt','r')
    data = f.read()
    lines = data.split('\n')[1:-1]
    
    X = []
    Y = []
    for l in lines:
        items = l.split('\t')
        w1 = embed(items[1].lower())
        w2 = embed(items[2].lower())
        dist = spatial.distance.cosine(w1,w2)
        """if np.isnan(dist):
            print('/',dist)
            print('/', items[0], len(items[0]))
            print('/', items[1], len(items[1]))
            print('/',w1)
            print('/',w2)
            print('/',items)
            input('>')"""
        X.append(dist)
        Y.append(float(items[0]))
        #print('%.4f'%(dist), items[2])

    corr, p_value=pearsonr(X,Y)
    print('/ant_syn r: ',corr)

    rho, p_value=spearmanr(X,Y)
    print('/ant_syn rho: ',rho)
    print('/')

def test_ant_syn_trios(n):
    f=open('evaluation_sets/ant_syn_trios.txt','r')
    data = f.read()
    lines = data.split('\n')
    
    start = 1000
    end = 11000
    if n != 10000:
        start = 0
        end = n

    orig_sentences, ant_sentences, syn_sentences = [], [], []
    for line in lines[start:end]:
        sentences = line.split('\t')
        orig_sentences.append(sentences[0])
        ant_sentences.append(sentences[1])
        syn_sentences.append(sentences[2])
    
    orig_vectors = list_embed(orig_sentences)   
    ant_vectors = list_embed(ant_sentences)   
    syn_vectors = list_embed(syn_sentences)

    count = 0
    num_valid = 0
    for o, a, s in zip(orig_vectors,ant_vectors,syn_vectors):
        dist1 = spatial.distance.cosine(o,a)
        dist2 = spatial.distance.cosine(o,s)
        num_valid += 1
        if dist2 < dist1:
            count += 1

    print('/ant_syn_trio distance n=' + str(n) + ': ' + str(100*float(count)/float(num_valid)) + '%')
    print('\n')

def test_neg_syn_trios(n):
    f=open('evaluation_sets/neg_syn_trios.txt','r')
    data = f.read()
    lines = data.split('\n')

    orig_sentences, neg_sentences, syn_sentences = [], [], []

    start = 1000
    end = 11000
    if n != 10000:
        start = 0
        end = n

    for line in lines[start:end]:
        sentences = line.split('\t')
        orig_sentences.append(sentences[0])
        neg_sentences.append(sentences[1])
        syn_sentences.append(sentences[2])
    
    orig_vectors = list_embed(orig_sentences)   
    neg_vectors = list_embed(neg_sentences)   
    syn_vectors = list_embed(syn_sentences)

    count = 0
    num_valid = 0
    for o, a, s in zip(orig_vectors,neg_vectors,syn_vectors):
        dist1 = spatial.distance.cosine(o,a)
        dist2 = spatial.distance.cosine(o,s)
        num_valid += 1
        if dist2 < dist1:
            count += 1

    print('/neg_syn_trio distance n=' + str(n) + ': ' + str(100*float(count)/float(num_valid)) + '%')
    print('\n')

print('\nimporting sample sentences for commonsense reasoning')
sys.stdout.flush()

def print_distance_measurements():
    test_sentence_pair('I am a cat','I am not a cat')
    test_sentence_pair('I am a cat','I am a domesticated cat')
    test_sentence_pair('I\'m ready for this','I\'m not ready for this')
    test_sentence_pair('I\'m ready for this','I am prepared for this')
    test_sentence_pair('That\'s bad','That\'s not bad')
    test_sentence_pair('That\'s bad','That\'s too bad')
    test_sentence_pair('No, don\'t play that song again','Play that song again')
    test_sentence_pair('Could you play that song again','Play that song again')
    test_sentence_pair('Delete that file','Don\'t delete that file')
    test_sentence_pair('Let\'s keep that file','Don\'t delete that file')
    test_sentence_pair('I\'m talking to you, Alexa','I wasn\'t talking to you, Alexa')
    test_sentence_pair('I\'m talking to you, Alexa', 'Did you hear me, Alexa')
    test_sentence_pair('That movie wasn\'t too bad','That movie was terrible')
    test_sentence_pair('That movie wasn\'t too bad','That movie was pretty good')
    test_sentence_pair('I want to make a reservation','I want to cancel my reservation')
    test_sentence_pair('I want to make a reservation','I would like to make a reservation')
    test_sentence_pair('That\'s fair','That\'s not fair')
    test_sentence_pair('That\'s fair','That seems fair to me')
    test_sentence_pair('I was born in California','I was not born in California')
    test_sentence_pair('I was born in California','I was born in San Francisco')


print('\n')
print('/distance measurements:')
print_distance_measurements()
sys.exit()

print('\n')
#test_ant_syn_trios(1000)
#test_neg_syn_trios(1000)
#test_ant_syn_trios(10000)
#test_neg_syn_trios(10000)

#print('\ntesting wordsim353')
#test_wordsim353()

#print('testing SimLex-999')
#test_simlex()

#print('testing sts_benchmark')
#test_sts_benchmark()

#print('testing entailment')
#test_entailment()

#print('testing ant_syn_sentences')
#test_ant_syn_sentences()

#input_file='data/context_prediction/pre-embedded_wikipedia_bert_sentences_10000_lines.txt'
input_file='data/sample_sentences_augmented2.txt'
ss, sv, gs, gv = get_sample_set(input_file)

print('\n\nGoogle results google words only')
#test_google(gs, gv)
#test_google(gs, gv,100)
#test_google(gs + ss, gv+sv,100)
#sys.exit()

# test semantic similarity
#run_quadcopter_test(n=3)

# test similarity metrics
#print('\n\n')
#test_similarity_metrics()

# test commonsense reasoning
#print('\nCommonsense reasoning')
#test_commonsense_reasoning()

#print('\n\n calculating SAT results')
test_SAT()

print('\n\n/Method was ' + METHOD)
print('/' + model)
