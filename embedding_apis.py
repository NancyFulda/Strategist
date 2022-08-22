import requests
import json
import numpy as np
import embeddings.embedding_client as client

#---------------------
# define embedding methods

def append_negations(sentences, vectors):
    vec_list = []
    for sentence,vector in zip(sentences, vectors):
        if ' not ' in sentence or \
           "n't " in sentence or \
            ' no ' in sentence or \
            'not' == sentence[:3] or \
            'not' == sentence[-3:]:
            negation = 1.0
        else:
            negation = 0.0
        new_vector = np.append(vector, negation)
        vec_list.append(new_vector)
    return np.array(vec_list)

def embed(sentences, method='use_lite', negations=False):

  if method == 'elmo':
    embedded = client.embed(sentences, encoder="Elmo", url="http://monster.cs.byu.edu:8081/invocations")
    if negations:
        return append_negations(sentences, embedded)
    else:
        return embedded

  elif method == 'use_lite':
    response = requests.post("http://rainbow.cs.byu.edu:8087/invocations",
                        json = {
                                "embed": {
                                    "sentences": sentences
                                    }
                                }
                        ).json()
    if negations:
        return append_negations(sentences, np.array(response['embed']['universal-sentence-encoder-lite']))
    else:
        try:
            return np.array(response['embed']['universal-sentence-encoder-lite'])
        except:
            print("Error embedding sentences ", sentences)
            print("API response", response)
            return np.zeros([len(sentences), 512])

  elif method == 'use_large':
    response = requests.post("http://candlelight.cs.byu.edu:8085/invocations",
                        json = {
                                "embed": {
                                    "sentences": sentences
                                    }
                                }
                        ).json()
    if 'embed' not in response:
        print("Some kind of embedding error has occurred:")
        print(response)
    if negations:
        return append_negations(sentences, np.array(response['embed']['universal-sentence-encoder-large']))
    else:
        return np.array(response['embed']['universal-sentence-encoder-large'])

  elif method == 'bag_of_words':
    response = requests.post("http://candlelight.cs.byu.edu:8085/invocations",
                        json = {
                                "embed": {
                                    "sentences": sentences,
                                    "negations": negations
                                    }
                                }
                        ).json()
    return np.array(response['embed']['fasttext-bag-of-words'])
  else:
    raise ValueError("Unknown embedding method: " + method)

