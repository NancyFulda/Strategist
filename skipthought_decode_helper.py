import penseur.penseur_utils as penseur_utils
import penseur.penseur
import numpy as np

class decode_helper:

    #def __init__(self, filename='ALPHA_data', penseur_model = None):
    #def __init__(self, filename='wikipedia_first_2000_lines', penseur_model = None):
    def __init__(self, filename='reddit', penseur_model = None):
#    def __init__(self, filename='larry_king_50000_lines', penseur_model = None):
#    def __init__(self, filename='larry_king_processed', penseur_model = None):
#    def __init__(self, filename='larry_king_processed_LONG', penseur_model = None):
        self.dec = penseur_utils.load_decoder(filename)
        if penseur_model == None:
            self.penseur = penseur.penseur.Penseur()
        else:
            self.penseur = penseur_model

    def encode(self, sentence):
        return self.penseur.get_vector(sentence)

    def decode(self, vector, num_sentence = 1):
        return penseur_utils.decode(self.dec, vector, 1)

    def average(self, sentences):
        num_vectors = float(len(sentences))
        v_sum = np.zeros([1,4800])
        for sentence in sentences:
            v = self.penseur.get_vector(sentence)
            v_sum += v
        v_sum = v_sum / num_vectors

        return self.decode(v_sum.astype(np.float32))

    def converse(self):

        print("ENTERING CONVERSATION MODE: type 'done' to exit.")
        sentence = raw_input("\nWhat shall we converse about? ")
        while (sentence != 'done'):
            v = self.penseur.get_vector(sentence)
            print(self.decode(v))
            sentence = raw_input(">")

    def jitter(self, sentence, stddev = 0.01, num_results = 10):
        #adds random amounts to the vector, as a way of sampling
        #the space near a given sentence
        if isinstance(sentence, basestring):
            v = self.penseur.get_vector(sentence)
        else:
            v = sentence

        jitter_results = []
        jitter_results.append(self.decode(v.astype(np.float32)))
        for i in range(num_results):
            v_prime = v + np.random.normal(0,stddev,len(v))
            jitter_results.append(self.decode(v_prime.astype(np.float32)))

        return jitter_results
