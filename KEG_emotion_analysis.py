import numpy as np
import sys, os
from os import walk
#import skyrim_parsing
import emotion_parsing
from scipy import spatial
import embedding_apis as embed_api
import random
import pickle as pkl
import dataset_fasttext
import SIF_embedding

#Original KEG file modified for IJCAI 2019...

#correct_categories = skyrim_parsing.build_from_file()
#human_captions = skyrim_parsing.get_human_labels()

#'human captions' is actually now just the text to be analyzed
correct_categories, human_captions = emotion_parsing.build_from_file()


categories = ['EMO_anger','EMO_disgust','EMO_fear', 'EMO_happiness','EMO_sadness', 'EMO_surprise']
#categories = ['KEG_threat','KEG_explore','KEG_barter']

baseline_words={}
baseline_words['EMO_anger'] = ['angry','frustrated','hate','can\'t','won\'t','stop']
baseline_words['EMO_disgust'] = ['ugh','gross','disgusting']
baseline_words['EMO_fear'] = ['np','wait','afraid']
baseline_words['EMO_happiness'] = ['happy','great','thank','glad','nice']
baseline_words['EMO_sadness'] = ['sad','sorry']
baseline_words['EMO_surprise'] = ['why','understand','don\'t']

#categorization_method = 'SKIPTHOUGHT'
#categorization_method = 'SKIPTHOUGHT_CENTROID'
#categorization_method = 'SKIPTHOUGHT_NEAREST'
#categorization_method = 'SPACY_CENTROID'
#categorization_method = 'SPACY_NEAREST'
#categorization_method = 'WORD2VEC_CENTROID'
#categorization_method = 'WORD2VEC_NEAREST'
#categorization_method = 'USE_LITE_CENTROID'
#categorization_method = 'USE_LITE_NEAREST'
#categorization_method = 'USE_LARGE_CENTROID'
#categorization_method = 'USE_LARGE_NEAREST'
#categorization_method = 'BAG_OF_WORDS_CENTROID'
#categorization_method = 'BAG_OF_WORDS_NEAREST'
#categorization_method = 'BERT1_CENTROID'
#categorization_method = 'BERT1_NEAREST'
#categorization_method = 'BERT2_CENTROID'
#categorization_method = 'BERT2_NEAREST'
#categorization_method = 'BSE_BERT_NEAREST'
categorization_method = 'BSE_embedding'
#categorization_method = 'GPT2_last'
#categorization_method = 'GPT2_avg'

BSE_WEIGHT_FILENAME='W0.pkl'

use_SIF = False
use_MEAN = False
precalculated_pc=False   #only relavent if use_SIF == True
SIF_pc = None
MEAN_pc = None

CUDA=True

if categorization_method in ['USE_LARGE_CENTROID','USE_LARGE_NEAREST']:
    import tensorflow as tf
    import tensorflow_hub as hub
    tf_embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

def use_large_encode(text):
    with tf.Session() as session:
        if isinstance(text, basestring):
            embeddings = tf_embed([text])
        else:
            # we assume it's a list of strings
            embeddings = tf_embed(text)
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return session.run(embeddings)

if categorization_method in ['GPT2_last','GPT2_avg']:
    if sys.version_info[0] >= 3:
        raise Exception("The GPT encoder only works in Python 2")

    import torch
    from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model

    # Load pre-trained model (weights)
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def gpt2_encode(text):
    # Encode some inputs
    indexed_tokens = tokenizer.encode(text.encode('utf-8'))

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        hidden_states, past = model(tokens_tensor)
        # past can be used to reuse precomputed hidden state in a subsequent predictions
        # (see beam-search examples in the run_gpt2.py example).

    hidden_states=hidden_states.reshape(-1,768)

    if categorization_method == 'GPT2_last':
        return hidden_states[-1].cpu().numpy()
    elif categorization_method == 'GPT2_avg':
        hidden_states=np.mean(hidden_states.cpu().numpy(),axis=0)
        return hidden_states

def bse_encode(text, bse_encoder, dataset):
    x = bse_encoder.ftext.get_indices(text)
    x = dataset.to_onehot(x)
    embedding = bse_encoder([x.cuda()]).data.cpu().numpy() if CUDA else bse_encoder([x]).data.numpy()
    #embedding = torch.cat([x_mu,x_sigma]).data.cpu().numpy()
    embedding = embedding.reshape(600)
    return embedding

def bert_encode(text, tokenizer, model, method):
    text = '[CLS] ' + text + ' [SEP]'
    tokens_tensor, segments_tensors = bert_embeddings.tokenize_text(unicode(text), tokenizer)
    encoded_layers = bert_embeddings.get_hidden_states(model, tokens_tensor, segments_tensors)

    if method == 'BERT1':
        layer = encoded_layers[-1]
    elif method == 'BERT2':
        layer = encoded_layers[-2]
    else:
        raise ValueError('Unknown encoding method for bert_encode: ' + method)

    return np.mean(np.squeeze(layer.cpu().numpy())[1:], axis=0)

if categorization_method in ['USE_LARGE_CENTROID','USE_LARGE_NEAREST']:
    print("Embedding use_large vectors")
    pre_embedded_vectors = {}
    sentence_list = [x.strip() for x in human_captions.values()]
    use_large_vectors = None
    print('len sentence list', len(sentence_list))
    chunks = [sentence_list[i*1000:(i+1)*1000] for i in range((len(sentence_list) + 1000 - 1) // 1000)]
    print('len chunkst', len(chunks))
    count = 1
    for chunk in chunks:
        print('embedding chunk ' + str(count) + ' of ' + str(len(chunks)))
        count += 1
        if use_large_vectors is None:
            #use_large_vectors = embed_api.embed(chunk, 'use_large')
            use_large_vectors = use_large_encode(chunk)
        else:
            #use_large_vectors = np.vstack([use_large_vectors,embed_api.embed(chunk, 'use_large')])
            use_large_vectors = np.vstack([use_large_vectors,use_large_encode(chunk)])
        print(use_large_vectors.shape)
    for i in range(len(sentence_list)):
        pre_embedded_vectors[sentence_list[i]] = use_large_vectors[i]

    print("Done")

print("Categorization method is ", categorization_method)
print("creating canonical examples")
if categorization_method in ['SKIPTHOUGHT','SKIPTHOUGHT_CENTROID','SKIPTHOUGHT_NEAREST']:
    print("Importing Strategist...")
    import strategist
    s=strategist.Strategist(penseur=True,scholar=False)
elif categorization_method in ['SPACY_CENTROID','SPACY_NEAREST']:
    import spacy
    nlp = spacy.load('en')
elif categorization_method in ['BERT1_CENTROID', 'BERT2_CENTROID', 'BERT1_NEAREST', 'BERT2_NEAREST']:
    import bert_embeddings
    from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

    # Load pre-trained model tokenizer (vocabulary)
    BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    BERT_model = BertModel.from_pretrained('bert-base-uncased')
    BERT_model.eval()
elif categorization_method in ['BSE_BERT_NEAREST']:
    import bert_embeddings
    from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

    # Load pre-trained model tokenizer (vocabulary)
    BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    BERT_model = BertModel.from_pretrained('bert-base-uncased')
    BERT_model.eval()

    f = open(BSE_WEIGHT_FILENAME, 'r')
    W = pkl.load(f)
elif categorization_method in ['BSE_embedding']:
    import torch

    # --ROUND 1--#
    #model = 'w2w_book_corpus_v1/w2w_encoder_760k.pkl'
    #model = 'w2w_book_corpus_v1/w2w_encoder_epoch_1_570k.pkl'
    #model = 'w2w_book_corpus_v1/encoder_epoch_1_820k.pkl'

    #model = 'autoencoder_book_corpus/encoder_epoch_5_300k.pkl'
    #model = 'phrase_plus_autoencoder_book_corpus/encoder_epoch_2_410k.pkl'
    #bse_encoder = torch.load('/mnt/pccfs/backed_up/nancy/archive_w2w_first_round_experiments/partially_trained_encoders/' + model)

    # --ROUND 2-- #

    #model = 'w2w_with_w2v_book_corpus/encoder_epoch_0_410k.pkl'
    #model = 'w2w_with_w2v_book_corpus/encoder_epoch_0_750k.pkl'
    #model = 'w2w_with_w2v_book_corpus/encoder_epoch_1_430k.pkl'

    #model = 'w2v_plus_autoencoder_book_corpus_trial2/encoder_epoch_0_730k.pkl'
    #model = 'w2v_plus_autoencoder_book_corpus_trial2/encoder_epoch_1_230k.pkl'

    #model = 'w2v_plus_autoencoder_book_corpus_encoder_no_fasttext/encoder_epoch_0_720k.pkl'
    #bse_encoder = torch.load('/mnt/pccfs/backed_up/nancy/archive_w2w_second_round_experiments/partially_trained_encoders/' + model)

    # *** THIRD ROUND *** #
    #model = 'neg_loss_plus_w2v_loss_book_corpus_no_fasttext/encoder_epoch_0_170k.pkl'
    #model = 'neg_loss_plus_w2v_loss_book_corpus_no_fasttext/encoder_epoch_0_300k.pkl'
    #model = 'neg_loss_plus_w2v_loss_book_corpus_no_fasttext/encoder_epoch_0_440k.pkl'

    #model = 'neg_plus_inv_loss_book_corpus_no_fasttext/encoder_epoch_0_130k.pkl'
    #model = 'neg_plus_inv_loss_book_corpus_no_fasttext/encoder_epoch_0_150k.pkl'
    #model = 'neg_plus_inv_loss_book_corpus_no_fasttext/encoder_epoch_0_200k.pkl'

    #model = 'no_inv_loss_book_corpus_no_fasttext/encoder_epoch_0_90k.pkl'
    #model = 'no_inv_loss_book_corpus_no_fasttext/encoder_epoch_0_120k.pkl'
    #model = 'no_inv_loss_book_corpus_no_fasttext/encoder_epoch_0_150k.pkl'

    #model = 'only_w2v_book_corpus_no_fasttext/encoder_epoch_0_780k.pkl'
    #model = 'only_w2v_book_corpus_no_fasttext/encoder_epoch_1_290k.pkl'

    #model = 'svi_plus_w2v_book_corpus_no_fasttext/encoder_epoch_0_310k.pkl'
    #model = 'svi_plus_w2v_book_corpus_no_fasttext/encoder_epoch_0_470k.pkl'

    #model = 'w2w_book_corpus_no_fasttext/encoder_epoch_0_240k.pkl'
    #model = 'w2w_book_corpus_no_fasttext/encoder_epoch_0_280k.pkl'

    #bse_encoder = torch.load('/mnt/pccfs/backed_up/nancy/archive_w2w_third_round_experiments/partially_trained_encoders/' + model)

    # *** THIRD ROUND *** #
    #model = 'output_autoencoder_book_corpus_baseline/3_encoder.pkl'
    #model = 'output_autoencoder_book_corpus_no_fasttext/3_encoder.pkl'
    #model = 'output_autoencoder_book_corpus_z_mu/9_encoder.pkl'
    #model = 'output_autoencoder_wikipedia/3_encoder.pkl'
    #model = 'output_autoencoder_wikipedia_no_fasttext/3_encoder.pkl'
    #model = 'output_w2v_book_corpus/4_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus/3_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_high_sample_freq/6_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_low_learning_rate/2_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_no_fasttext/6_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_z_mu/3_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_wikipedia/3_encoder.pkl'
    #model = 'output_w2v_wikipedia/1_encoder.pkl'
    #model = 'output_w2v_wikipedia_no_fasttext/4_encoder.pkl'
    
    #bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v5/' + model)

    # *** FIFTH ROUND *** #
    #model = 'output_autoencoder_book_corpus_baseline/3_encoder.pkl'
    #model = 'output_autoencoder_book_corpus_no_fasttext/3_encoder.pkl'
    #model = 'output_autoencoder_book_corpus_z_mu/6_encoder.pkl'
    #model = 'output_autoencoder_wikipedia/3_encoder.pkl'
    #model = 'output_autoencoder_wikipedia_no_fasttext/3_encoder.pkl'
    #model = 'output_w2v_book_corpus/9_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus/23_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_high_sample_freq/6_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_low_learning_rate/2_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_no_fasttext/6_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_book_corpus_z_mu/3_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_wikipedia/3_encoder.pkl'
    #model = 'output_w2v_wikipedia/3_encoder.pkl'
    #model = 'output_w2v_wikipedia_no_fasttext/4_encoder.pkl'

    #bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v5/' + model)

    # ** Sixth Round **
    #model = 'output_w2v_book_corpus_z_mu/156_encoder.pkl'
    #model = 'output_w2v_wikipedia_z_mu/265_encoder.pkl'
    #model = 'output_w2v_book_corpus_z_mu_higher_learning_rate/57_encoder.pkl'
    #model = 'output_w2v_plus_neg_wikipedia_z_mu/38_encoder.pkl'
    model = 'output_autoencoder_plus_neg_wikipedia_z_mu/22_encoder.pkl'
    #model = 'output_autoencoder_plus_neg_wikipedia_z_mu_trial2/16_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_plus_neg_wikipedia_z_mu_trial2/4_encoder.pkl'
    #model = 'output_w2v_plus_autoencoder_split_corpus_1_z_mu/15_encoder.pkl'

    bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v6/' + model)

    # ** Seventh Round **
    #model = 'output_autoencoder_book_corpus/30_encoder.pkl'
    #model = 'output_context_plus_inv_text_book_corpus/8_encoder.pkl'
    #model = 'output_context_plus_inv_text_book_corpus_trial2/17_encoder.pkl'
    #model = 'output_context_plus_w2v_book_corpus/16_encoder.pkl'
    #model = 'output_context_plus_w2v_plus_inv_text_book_corpus/14_encoder.pkl'
    #model = 'output_context_plus_w2v_plus_inv_text_book_corpus_higher_context_loss/5_encoder.pkl'
    ##model = 'output_context_plus_w2v_split_corpus_1/0_encoder.pkl'
    ##model = 'output_inv_text_plus_autoencoder_book_corpus/0_encoder.pkl'
    ##model = 'output_inv_text_plus_w2v_book_corpus/0_encoder.pkl'
    #model = 'output_w2v_book_corpus/195_encoder.pkl'
    #model = 'output_w2v_plus_context_book_corpus/4_encoder.pkl'

    #bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v7/' + model)

    # ** Eigth Round **
    #model = 'output_context_book_corpus/8_encoder.pkl'
    #model = 'output_context_book_corpus_bigger_learning_rate/7_encoder.pkl'
    ##model = 'output_context_plus_w2v_split_corpus_1/0_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_1_w2v_overdrive_10/31_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_1_w2v_overdrive_100/79_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_2/3_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_2_w2v_overdrive_10/29_encoder.pkl'
    #model = 'output_context_plus_w2v_split_corpus_2_w2v_overdrive_100/69_encoder.pkl'
    #model = 'output_inv_text_plus_autoencoder_book_corpus/2_encoder.pkl'
    #model = 'output_inv_text_plus_context_book_corpus/3_encoder.pkl'
    ##model = 'output_inv_words_plus_autoencoder_book_corpus/0_encoder.pkl'
    ##model = 'output_inv_words_plus_context_book_corpus/0_encoder.pkl'

    #bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v8/' + model)


    # ** Ninth Round **
    #model = 'output_context_plus_w2v_split_corpus_2_max_len_512/15_encoder.pkl'
    #model = 'output_autoencoder_book_corpus_max_len_512/6_encoder.pkl'
    #model = 'output_context_plus_reconstruction_book_corpus/32_encoder.pkl'
    #model = 'output_context_plus_reconstruction_book_corpus_max_len_512/10_encoder.pkl'

    #bse_encoder = torch.load('/mnt/pccfs/not_backed_up/nancy/docker/better_sentence_embeddings_v9/' + model)


    dataset=dataset_fasttext.Dataset()
elif categorization_method in ['USE_LITE_CENTROID','USE_LITE_NEAREST','USE_LARGE_NEAREST','USE_LARGE_CENTROID','BAG_OF_WORDS_CENTROID','BAG_OF_WORDS_NEAREST','ELMO_CENTROID','ELMO_NEAREST','GPT2_last','GPT2_avg']:
    pass
else:
    raise ValueError('Unknown categorization method: ' + categorization_method)

if categorization_method in ['SKIPTHOUGHT','SKIPTHOUGHT_CENTROID','SKIPTHOUGHT_NEAREST']:
    canonical_examples = {}
    canonical_vectors = {}
    canonical_centroids = {}
    for c in categories:
        #s.import_canonical_examples(c,'penseur')
        canonical_examples[c] = []
        canonical_vectors[c] = []
        for line in open('canonical_examples/' + c + '.txt', 'r').read().strip().split('\n'):
            text = line.split(' :: ')[1].strip()
            canonical_examples[c].append(text.decode('utf8'))
            canonical_vectors[c].append(s.encode(text.decode('utf8')))
        canonical_centroids[c] = np.mean(np.array(canonical_vectors[c]),0)
elif categorization_method in['SPACY_CENTROID','SPACY_NEAREST','USE_LITE_CENTROID','USE_LITE_NEAREST','USE_LARGE_NEAREST', 'USE_LARGE_CENTROID','BAG_OF_WORDS_CENTROID','BAG_OF_WORDS_NEAREST','BERT1_NEAREST','BERT2_NEAREST','BERT1_CENTROID','BERT2_CENTROID','BSE_BERT_NEAREST','BSE_embedding','GPT2_last','GPT2_avg']:
    canonical_examples = {}
    canonical_vectors = {}
    canonical_centroids = {}
    for c in categories:
        canonical_examples[c] = []
        canonical_vectors[c] = []
        use_large_sentences = []
        for line in open('canonical_examples/' + c + '.txt', 'r').read().strip().split('\n'):
            try:
                line = unicode(line.decode('utf8'))
            except:
                line = str(line)
            text = line.split(' :: ')[1].strip()
            if categorization_method in ['SPACY_CENTROID','SPACY_NEAREST']:
                text_vector = nlp(unicode(text)).vector
            elif categorization_method in ['USE_LITE_CENTROID','USE_LITE_NEAREST']:
                text_vector = embed_api.embed([text], 'use_lite')[0]
            elif categorization_method in ['USE_LARGE_CENTROID', 'USE_LARGE_NEAREST']:
                #text_vector = embed_api.embed([text], 'use_large')[0]
                use_large_sentences.append(text)
                text_vector = None
            elif categorization_method in ['BAG_OF_WORDS_CENTROID','BAG_OF_WORDS_NEAREST']:
                text_vector = embed_api.embed([text], 'bag_of_words')[0]
            elif categorization_method in ['BERT1_NEAREST','BERT1_CENTROID']:
                text_vector = bert_encode(text, BERT_tokenizer, BERT_model, 'BERT1')
            elif categorization_method in ['BERT2_CENTROID','BERT2_NEAREST']:
                text_vector = bert_encode(text, BERT_tokenizer, BERT_model, 'BERT2')
            elif categorization_method in ['BSE_BERT_NEAREST']:
               bert_embedding = bert_encode(text, BERT_tokenizer, BERT_model, 'BERT1')
               text_vector = np.sum(np.multiply(bert_embedding,W.T),axis=1)
            elif categorization_method in ['BSE_embedding']:
               text_vector = bse_encode(text, bse_encoder, dataset)
            elif categorization_method in ['GPT2_last','GPT2_avg']:
               text_vector = gpt2_encode(text)
            #canonical_examples[c].append(nlp(text))
            canonical_examples[c].append(text)
            #canonical_vectors[c].append(nlp(text).vector)
            canonical_vectors[c].append(text_vector)
        if categorization_method in ['USE_LARGE_CENTROID','USE_LARGE_NEAREST']:
            #canonical_vectors[c] = embed_api.embed(use_large_sentences, 'use_large')
            canonical_vectors[c] = use_large_encode(use_large_sentences)
        canonical_centroids[c] = np.mean(np.array(canonical_vectors[c]),0)
else:
    raise ValueError('Unknown categorization method: ' + categorization_method)

#print "Import complete!"
print("done")

if (use_SIF or use_MEAN):
    if precalculated_pc:
        f=open('SIF_pc_wikipedia_bert.txt')
        line = f.readline().strip('\n')
        SIF_pc = [float(x) for x in line.split()]
        SIF_pc = np.stack(SIF_pc)
    else:
        print('Calculating principle components...')
        X = []

        #find the principle components
        for c in categories:
            for v in canonical_vectors[c]:
                X.append(v)
        if use_SIF:
            SIF_pc = SIF_embedding.compute_pc(X)
            #subtract principle components from all example vectors
            for c in categories:
                for idx, v in enumerate(canonical_vectors[c]):
                    SIF_pc = SIF_pc.reshape([1,-1])
                    canonical_vectors[c][idx] = v-v.dot(SIF_pc.T) * SIF_pc
        if use_MEAN:
            MEAN_pc = np.mean(X, axis=0)
            #subtract mean from all example vectors
            for c in categories:
                for idx, v in enumerate(canonical_vectors[c]):
                    canonical_vectors[c][idx] = canonical_vectors[c][idx]-MEAN_pc


def categorize(text):
    if categorization_method == 'SKIPTHOUGHT':
        raise ValueError("Method SKIPTHOUGHT method no longer supported")
        return s.categorize(s.encode(text), category, mode='skipthought')
    elif categorization_method == 'SKIPTHOUGHT_CENTROID':
        doc_vector = s.encode(text)
        if use_SIF:
            #doc_vector = SIF_embedding.remove_pc(np.array([doc_vector]), npc=1, pc=SIF_pc)
            v=doc_vector.copy()
            doc_vector = v-v.dot(SIF_pc.T) * SIF_pc
        if use_MEAN:
            doc_vector = doc_vector - MEAN_pc

        centroid_similarities = {}
        distances = []
        cat_vector = []
        for c in categories:
            #centroid_similarities[c] = spatial.distance.cosine(doc_vector, canonical_centroids[c])
            #distances.append(1-centroid_similarities[c])
            #cat_vector.append((1-centroid_similarities[c]) < 0.75)
            
            midpoint0 = canonical_centroids[c]
            midpoint1 = np.zeros(len(canonical_centroids[c]))
            for c2 in categories:
                if c2 != c:
                    midpoint1 += canonical_centroids[c2]
            midpoint1 = midpoint1/(len(categories)-1)
            dist0 = spatial.distance.cosine(doc_vector, midpoint0)
            dist1 = spatial.distance.cosine(doc_vector, midpoint1)

            if dist0 > dist1:
                cat_vector.append(False)
            else:
                cat_vector.append(True)

        #val = max(centroid_similarities, key=centroid_similarities.get)
        val = None

        #test
        #cat_vector=[0,0,0,0]
        #cat_vector[categories.index(val)] = True
        return val, cat_vector, distances
    elif categorization_method == 'SKIPTHOUGHT_NEAREST':
        doc_vector = s.encode(text)
        if use_SIF:
            #doc_vector = SIF_embedding.remove_pc(np.array([doc_vector]), npc=1, pc=SIF_pc)
            v=doc_vector.copy()
            doc_vector = v-v.dot(SIF_pc.T) * SIF_pc
        if use_MEAN:
            doc_vector = doc_vector - MEAN_pc

        nearest_category = None
        nearest_distance = 1e12
        for c in categories:
            for e in canonical_examples[c]:
                dist = spatial.distance.cosine(doc_vector, s.encode(e))
                if dist < nearest_distance:
                    nearest_category = c
                    nearest_distance = dist
        cat_vector=[0,0,0,0,0,0]
        cat_vector[categories.index(nearest_category)] = True
        return nearest_category, cat_vector, None
    elif categorization_method  in ['SPACY_CENTROID', 'USE_LITE_CENTROID', 'USE_LARGE_CENTROID','BAG_OF_WORDS_CENTROID','BERT1_CENTROID','BERT2_CENTROID']:
        if categorization_method == 'SPACY_CENTROID':
            doc_vector = nlp(unicode(text)).vector
        elif categorization_method == 'USE_LITE_CENTROID':
            doc_vector = embed_api.embed([text], 'use_lite')[0]
        elif categorization_method == 'USE_LARGE_CENTROID':
            #doc_vector = embed_api.embed([text], 'use_large')[0]
            doc_vector = pre_embedded_vectors[text]
        elif categorization_method == 'BAG_OF_WORDS_CENTROID':
            doc_vector = embed_api.embed([text], 'bag_of_words')[0]
        elif categorization_method == 'BERT1_CENTROID':
            doc_vector = bert_encode(text, BERT_tokenizer, BERT_model, 'BERT1')
        elif categorization_method == 'BERT2_CENTROID':
            doc_vector = bert_encode(text, BERT_tokenizer, BERT_model, 'BERT2')
        if use_SIF:
            #doc_vector = SIF_embedding.remove_pc(np.array([doc_vector]), npc=1, pc=SIF_pc)
            v=doc_vector.copy()
            doc_vector = v-v.dot(SIF_pc.T) * SIF_pc
        if use_MEAN:
            doc_vector = doc_vector - MEAN_pc


        centroid_similarities = {}
        distances = []
        cat_vector = []
        for c in categories:
            #centroid_similarities[c] = spatial.distance.cosine(doc_vector, canonical_centroids[c])
            #distances.append(1-centroid_similarities[c])
            #cat_vector.append((1-centroid_similarities[c]) < 0.75)

            midpoint0 = canonical_centroids[c]
            midpoint1 = np.zeros(len(canonical_centroids[c]))
            for c2 in categories:
                if c2 != c:
                    midpoint1 += canonical_centroids[c2]
            midpoint1 = midpoint1/(len(categories)-1)
            #dist0 = spatial.distance.cosine(doc_vector, midpoint0)
            #dist1 = spatial.distance.cosine(doc_vector, midpoint1)
            #dist0 = spatial.distance.cityblock(doc_vector, midpoint0)
            #dist1 = spatial.distance.cityblock(doc_vector, midpoint1)
            dist0 = spatial.distance.euclidean(doc_vector, midpoint0)
            dist1 = spatial.distance.euclidean(doc_vector, midpoint1)

            if dist0 > dist1:
                cat_vector.append(False)
            else:
                cat_vector.append(True)

        #val = max(centroid_similarities, key=centroid_similarities.get)
        val = None

        return val, cat_vector, distances
    elif categorization_method in ['SPACY_NEAREST', 'USE_LITE_NEAREST','USE_LARGE_NEAREST','BAG_OF_WORDS_NEAREST','BERT1_NEAREST','BERT2_NEAREST','BSE_BERT_NEAREST','BSE_embedding','GPT2_last','GPT2_avg']:
        if categorization_method == 'SPACY_NEAREST':
            doc_vector = nlp(unicode(text)).vector
        elif categorization_method == 'USE_LITE_NEAREST':
            doc_vector = embed_api.embed([text], 'use_lite')[0]
        elif categorization_method == 'USE_LARGE_NEAREST':
            #doc_vector = embed_api.embed([text], 'use_large')[0]
            doc_vector = pre_embedded_vectors[text]
        elif categorization_method == 'BAG_OF_WORDS_NEAREST':
            doc_vector = embed_api.embed([text], 'bag_of_words')[0]
        elif categorization_method == 'BERT1_NEAREST':
            doc_vector = bert_encode(text, BERT_tokenizer, BERT_model, 'BERT1')
        elif categorization_method == 'BERT2_NEAREST':
            doc_vector = bert_encode(text, BERT_tokenizer, BERT_model, 'BERT2')
        elif categorization_method == 'BSE_BERT_NEAREST':
            bert_embedding = bert_encode(text, BERT_tokenizer, BERT_model, 'BERT1')
            doc_vector = np.sum(np.multiply(bert_embedding,W.T),axis=1)
        elif categorization_method == 'BSE_embedding':
            doc_vector = bse_encode(text, bse_encoder, dataset)
        elif categorization_method in ['GPT2_last','GPT2_avg']:
            doc_vector = gpt2_encode(text)
        if use_SIF:
            #doc_vector = SIF_embedding.remove_pc(np.array([doc_vector]), npc=1, pc=SIF_pc)
            v=doc_vector.copy()
            doc_vector = v-v.dot(SIF_pc.T) * SIF_pc
        if use_MEAN:
            doc_vector = doc_vector - MEAN_pc


        nearest_category = None
        nearest_distance = 1e20
        for c in categories:
            for v in canonical_vectors[c]:
                #dist = spatial.distance.cosine(doc_vector, v)
                #dist = spatial.distance.cityblock(doc_vector, v)
                dist = spatial.distance.euclidean(doc_vector, v)
                if dist < nearest_distance:
                    nearest_category = c
                    nearest_distance = dist
        cat_vector=[0,0,0,0,0,0]
        cat_vector[categories.index(nearest_category)] = True
        return nearest_category, cat_vector, None
    else:
        raise ValueError('Unknown categorization method: ' + categorization_method)

print("collecting skyrim files")
#basedir='Skyrim dataset combined small'
#basedir='Skyrim dataset combined'
basedir='Skyrim dataset images with human captions'
#filenames=os.listdir(basedir)
#print filenames

human_num_correct=0
human_num_false=0
human_num_exact_matches=0

human_baseline_num_correct=0
human_baseline_num_false=0
human_baseline_num_exact_matches=0

false_positives = [0,0,0,0,0,0]
failed_detections = [0,0,0,0,0,0]

baseline_false_positives = [0,0,0,0,0,0]
baseline_failed_detections = [0,0,0,0,0,0]

from tqdm import tqdm

print("beginning main program loop")

count = 0
for f in tqdm(human_captions.keys()):
    count+=1
    #print(count)

    for m in ['skipthought']:

        text = human_captions[f].strip()
        #print 'human label: ' + text
        
        output_string = 'CATEGORIES: '
        
        val, cat_vector, distances = categorize(text)
        #print('val is ', val)
        #print('cat_vector is ', cat_vector)

        for i in range(len(categories)):
            if correct_categories[f][i] == cat_vector[i]:
                human_num_correct += 1
            else:
                human_num_false += 1
                if cat_vector[i] == True:
                    false_positives[i] += 1
                else:
                    failed_detections[i] += 1

        human_boolean_cats=[0,0,0,0,0,0]
        human_baseline_categories=[0,0,0,0,0,0]

        #when you want a random baseline...
        r = random.randint(0,5)
        if r == 0:
            human_baseline_categories[0] = True
        elif r == 1:
            human_baseline_categories[1] = True
        elif r == 2:
            human_baseline_categories[2] = True
        elif r == 3:
            human_baseline_categories[3] = True
        elif r == 4:
            human_baseline_categories[4] = True
        else:
            human_baseline_categories[5] = True

        for i in range(len(categories)):
            #for word in baseline_words[categories[i]]:
            #    if word in text.split(' '):
            #        human_baseline_categories[i] = True
            
            #when you want a random baseline...
            #if random.randint(0,1) == 0:
            #    human_baseline_categories[i] = True
            #else:
            #    human_baseline_categories[i] = False

            if human_baseline_categories[i] == correct_categories[f][i]:
                human_baseline_num_correct += 1
            else:
                human_baseline_num_false += 1
                if human_baseline_categories[i] == True:
                    baseline_false_positives[i] += 1
                else:
                    baseline_failed_detections[i] += 1
        
        #check for exact matches
        if cat_vector is not None:
            if cat_vector == correct_categories[f]:
                #print "MATCH"
                human_num_exact_matches += 1
        if human_baseline_categories == correct_categories[f]:
            human_baseline_num_exact_matches += 1

        #print output_string + ", correct answer: " + str(correct_categories[f]) + '\n'
        #print 'baseline with human labels:' + str(human_baseline_categories)

print('\n\n')
print("OVERALL RESULTS: \n'")

print("Human:")
print("%d correct, %d false, %f percent" % (human_num_correct, human_num_false, 100.0 * float(human_num_correct)/float(human_num_correct + human_num_false)))
print("%d exact matches, %f percent" % (human_num_exact_matches, 100.0*human_num_exact_matches/len(human_captions.keys())))
print('\n')

print("Baseline using human text:")
print("%d correct, %d false, %f percent" % (human_baseline_num_correct, human_baseline_num_false, 100.0 * float(human_baseline_num_correct)/float(human_baseline_num_correct + human_baseline_num_false)))
print("%d exact matches, %f percent" % (human_baseline_num_exact_matches, 100.0*human_baseline_num_exact_matches/len(human_captions.keys())))
print('\n')

print("False positives:")
print(false_positives)
print("Failed detections:")
print(failed_detections)
print('\n')

print("Baseline False positives:")
print(baseline_false_positives)
print("Baseline Failed detections:")
print(baseline_failed_detections)
print('\n')

print(categorization_method)
if use_SIF:
    print('SIF')
if use_MEAN:
    print('MEAN')
print(model)
