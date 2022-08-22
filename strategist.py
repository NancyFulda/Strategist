import scholar.scholar as sch
import penseur.penseur as pens
#import hyperangles
from scipy import spatial
import skipthought_decode_helper 
import numpy as np
from scipy import spatial
import sys
import pickle
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub

# from ben
#no basestring in python3 Defining:
if (sys.version_info[0] >= (3.0)):
	basestring = (str,bytes)




class Strategist:

    def __init__(self, slim=True, penseur=True, scholar=True, universal=False, GoogleNews=False, use_angles=False, corl=False, intentional_agent=False, ia_recalculate=False):
        self.use_angles = use_angles
        self.metric = 'cosine'
        self.stddev = 0.01
        self.desired_dimensions=[]
        #self.alpha=1.0
        self.alpha=0.5
        self.GoogleNews = GoogleNews
        self.import_directory = 'canonical_examples/'

        self.cached_encoding_mode = ''
        self.cached_encodings={}
        self.cashed_word_encodings={}

        if penseur == True:
            self.penseur = pens.Penseur()
            self.decode_helper = skipthought_decode_helper.decode_helper('NIPS9', self.penseur)

        if scholar == True:
            if self.GoogleNews==True:
                self.scholar = sch.Scholar(GoogleNews)
            else:
                self.scholar = sch.Scholar(slim=slim) #slim=True for smaller corpus for faster search

        self.universal = False
        if universal == True:
            self.universal = True
            self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
            tf.logging.set_verbosity(tf.logging.ERROR)
            self.session = tf.Session()
            self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        ##! WOAH NELLY... for some reason importing during construction
        ##! makes it go WAAAAAAY slow. so we'll just make a separate
        ##! function for the section below...

        #load some base sentences into Penseur
        #f = open(penseur_input, 'r')
        #sentences=[]
        #for line in f:
            #    sentences.append(line.strip(' \n'))
        #self.penseur.encode(sentences)

        self.canonical_examples = {}
        self.canonical_tag_types = {}
        self.canonical_vectors = {}
        self.canonical_pisa_alphas = {}
        self.canonical_pisa_alphas['empty item']=''
        self.canonical_centroids = {}
        #self.canonical_target_points = {}
        self.canonical_angles = {}
        self.canonical_scale_factors = {}
        self.scale_vectors = {}
        self.context_vectors = {}

        if scholar == True:
            if corl == True:
                self.import_directory = 'CoRL_canonical_examples/'
                self.import_canonical_examples('accessing_containers', 'scholar')
                self.import_canonical_examples('affordance', 'scholar')
                self.import_canonical_examples('belong', 'scholar')
                self.import_canonical_examples('containers', 'scholar')
                self.import_canonical_examples('ownership', 'scholar')
                self.import_canonical_examples('rooms_for_locations', 'scholar')
                self.import_canonical_examples('rooms_for_objects', 'scholar')
                self.import_canonical_examples('tools', 'scholar')
                self.import_canonical_examples('trash_or_treasure', 'scholar')
                self.import_canonical_examples('word_travel', 'scholar')
            elif intentional_agent == True:
                #Do nothing - the scholar imports are handled under the 
                #penseur flag, so they'll be built into the pkl file
                pass                
            else:
                self.import_canonical_examples('affordance', 'scholar')
                self.import_canonical_examples('hypernyms', 'scholar')
                self.import_canonical_examples('meronyms', 'scholar')
                """
                self.import_canonical_examples('inverse_affordance', 'scholar')
                self.import_canonical_examples('word_travel', 'scholar')
                self.import_canonical_examples('gender', 'scholar')
                self.import_canonical_examples('gender2', 'scholar')
                self.import_canonical_examples('inverse_gender', 'scholar')
                self.import_canonical_examples('antonyms', 'scholar')
                self.import_canonical_examples('noun_antonyms', 'scholar')
                self.import_canonical_examples('verb_antonyms', 'scholar')
                """
        if penseur == True:
            self.import_canonical_examples('drone_navigation','penseur')
            self.drone_examples = ['launch','land','forward','backward','stabilize','tilt left','tilt right','turn left','turn right']
            self.drone_vectors = []
            for example in self.drone_examples:
                self.drone_vectors.append(self.encode(example))
            #self.import_canonical_examples('simon_invoked','penseur')
            #self.import_canonical_examples('simon_question','penseur')
            #self.import_canonical_examples('simon_math','penseur')
            #self.import_canonical_examples('simon_math2','penseur')
            #self.import_canonical_examples('simon_math3','penseur')
            if intentional_agent==True:
                if ia_recalculate == True:
                    self.import_canonical_examples('ia_location','penseur')
                    self.import_canonical_examples('ia_action_feedback','penseur')
                    self.import_canonical_examples('ia_object_info','penseur')
                    self.import_canonical_examples('ia_objects_of_interest','penseur')
                    self.import_canonical_examples('ia_state_transition','penseur')
                    self.import_canonical_examples('ia_potential_locations','penseur')
                    self.import_canonical_examples('ia_update_inventory','penseur')
                    self.import_canonical_examples('ia_environment_info','penseur')
                    self.import_canonical_examples('ia_good_prepositions', 'penseur')
                    self.import_canonical_examples('ia_invalid_action', 'penseur')
                    self.import_canonical_examples('ia_preposition_indicated', 'penseur')
                    self.import_canonical_examples('ia_navigation_failed', 'penseur')
                    self.import_canonical_examples('ia_verb_prep', 'penseur')
                    self.import_canonical_examples('ia_threat', 'penseur')
                    #self.import_canonical_examples('ia_victory', 'penseur')
                    #self.import_canonical_examples('ia_defeat', 'penseur')
                    self.import_canonical_examples('ia_logical_verbnoun', 'penseur')
                    self.import_canonical_examples('affordance', 'scholar')
                    self.import_canonical_examples('facilitates', 'scholar')
                    self.import_canonical_examples('belong', 'scholar')
                    self.import_canonical_examples('manipulability', 'scholar')
                    self.import_canonical_examples('verb_for_goal', 'penseur')
                    self.import_canonical_examples('subgoal_for_goal', 'penseur')
                    self.import_canonical_examples('object_for_goal_specific', 'penseur')
                    self.import_canonical_examples('object_for_goal_test', 'penseur')
                    self.import_canonical_examples('adjectives','penseur')

                    f=open('ia_canonical_data.pkl','wb')
                    pickle.dump(self.canonical_tag_types,f)
                    pickle.dump(self.canonical_vectors,f)
                    pickle.dump(self.canonical_centroids,f)
		    pickle.dump(self.canonical_pisa_alphas,f)
                    f.close()
                else:
                    f=open('ia_canonical_data.pkl','rb')
                    if sys.version_info[0] >= (3.0):
                        self.canonical_tag_types = pickle.load(f, encoding='latin1')
                        self.canonical_vectors = pickle.load(f, encoding='latin1')
                        self.canonical_centroids = pickle.load(f, encoding='latin1')
                        self.canonical_pisa_alphas = pickle.load(f, encoding='latin1')
                    else:
                        self.canonical_tag_types = pickle.load(f)
                        self.canonical_vectors = pickle.load(f)
                        self.canonical_centroids = pickle.load(f)
                        self.canonical_pisa_alphas = pickle.load(f)
                    f.close()
            else:
                """self.import_canonical_examples('causation', 'penseur')
                self.import_canonical_examples('travel', 'penseur')
                self.import_canonical_examples('how_do_you_X', 'penseur')
                self.import_canonical_examples('how_do_you_X2', 'penseur')
		"""

    def create_context(self, data_points, scale = 1.9):

	if isinstance(data_points[0],basestring):
	    vectors = []
	    for p in data_points:
		#assume that we've been handed a list of words
		#also assume that we're using scholar right now
		vectors.append(self.word_encode(self.tag(p)))
	else:
	    vectors = data_points
	    
	
	#note that scale vector returns
	#1-variance along each dimension
	context = self.scale_vector(vectors)
	
	#translate to centered-around-zero
	context = context - np.mean(context)

	#scale to a prespecified range
	#(We want max-min to equal ~ 1.5)
	max_val = np.max(context)
	min_val = np.min(context)
	cur_range = max_val - min_val
	desired_scale = scale/cur_range

	#ALTERNATE THING TO TRY
	#Raise translated (mean at 1) context to power instead
	#of scaling its distance around zero...

	context = context * desired_scale
	#translate to be centered-around-one
	context = context + 1
	#return
	return context


    def load_context(self, context_name, scale=1.9):
	f=open('contexts/' + context_name + '.txt','r')
	words=[]
	v_words = []
	for line in f:
	    words.append(line.strip())
	    v_words.append(self.word_encode(line.strip()))

	self.context_vectors[context_name] = self.create_context(v_words,scale)
	return self.context_vectors[context_name]

	    
    def import_penseur_sentences(self,penseur_input='skipthought_test_sentences.txt', limit=1000):
        if self.penseur.vectors != None:
            print("WARNING: Currently encoded vectors in penseur are now being overwritten...")

        #load some base sentences into Penseur...
        f = open(penseur_input, 'r')
        print("Importing " + str(limit) + " sentences from " + penseur_input)
        #sentences = f.read().split('\n')
        #for s in sentences[:limit]:
            #self.encode(s.strip(' '))
        sentences=[]
        for line in f:
            if len(line) > 2:
                sentences.append(line.strip(' \n'))
        self.penseur.encode(sentences[:limit])

    def import_ia_agent_sentences(self, sentences):
        self.penseur.encode(sentences)

    def import_canonical_examples(self, tag, model_name='', text_file=''):
        if text_file == '':
            text_file = self.import_directory + tag + '.txt'
        f = open(text_file, 'r')

        #if no model is given, then infer the model from the tag
        if model_name == '':
            model_name = self.model_name(tag)
        else:
            if tag not in self.canonical_tag_types:
                self.canonical_tag_types[tag] = model_name
            else:
                if self.canonical_tag_types[tag] != model_name:
                    print("ERROR in Strategist.import_canonical_examples")
                    print("canonical_tag_types lists 'tag' as " + self.canonical_tag_types[tag] + ", but the given model has type " + model_name)


        #PENSEUR MODEL
        if model_name == 'penseur':
            for line in f:
                sentences = line.split('::')
                if tag not in self.canonical_examples.keys():
                     self.canonical_examples[tag] = []
                self.canonical_examples[tag].append([s.strip('\n ') for s in sentences])
        
        #SCHOLAR MODEL
        elif model_name == 'scholar':
            for line in f:
                words = line.split(' ')
                if tag not in self.canonical_examples.keys():
                     self.canonical_examples[tag] = []
                self.canonical_examples[tag].append([w.strip('\n ') for w in words])

        else:
            raise ValueError("ERROR: unknown data type for Strategist.import_canonical_examples: " + model_name)

        #Now that all the examples have been loaded in, calculate the
        #canonical vector (or angle) for this relation
        self.calculate_canonical_vector(tag)
	#self.calculate_canonical_pisa_alphas(tag)
        if model_name == 'scholar' and self.use_angles == True:
            self.calculate_canonical_angles(tag)

        #calculate the scale factors for tag-specific distance metrics
        self.scale_vectors[tag] = self.tag_scale(tag)


    def calculate_canonical_vector(self, tag):
        if tag not in self.canonical_tag_types.keys():
            raise ValueError("Tag not found in self.canonical_tag_types")
        model_name = self.canonical_tag_types[tag]

        print("Creating canonical vectors for tag: " + tag)

        if model_name == 'scholar':
            if tag not in self.canonical_examples.keys():
                raise ValueError("Tag '" + tag + "' not in self.canonical_examples.")
            sum_vector = 0
            #target_point = 0
            examples = self.canonical_examples[tag]
            divide_by = 0 #count how many we actually use...
            
            for i in range(len(examples)):
                if self.scholar.exists_in_model(examples[i][1]) and self.scholar.exists_in_model(examples[i][0]):
                    #vector = self.scholar.model[examples[i][1]] - self.scholar.model[examples[i][0]]
                    vector = self.word_encode(examples[i][1]) - self.word_encode(examples[i][0])
                    sum_vector += vector
                    #target_point += self.word_encode(examples[i][1])
                    divide_by += 1
            else:
                if not self.scholar.exists_in_model(examples[i][0]):
                    print("   *** Word " + examples[i][0] + " does not exist in the scholar model. Omitting...")
                if not self.scholar.exists_in_model(examples[i][1]):
                    print("   *** Word " + examples[i][1] + " does not exist in the scholar model. Omitting...")

            if divide_by == 0:
                raise ValueError("It looks like your input file for '" + tag + "' didn't parse properly. Check that you used tagged values for each word, eg house_NN")
            self.canonical_vectors[tag] = sum_vector/divide_by
            #self.canonical_target_points[tag] = target_point/divide_by

        elif model_name == 'penseur':
            if tag not in self.canonical_examples.keys():
                raise ValueError("Tag '" + tag + "' not in self.canonical_examples.")
            sum_vector = 0
            divide_by = 0
            examples = self.canonical_examples[tag]
            centroid0 = 0
            centroid1 = 0
            for i in range(len(examples)):
                if len(examples[i]) != 2:
                    raise ValueError("Your input file for '" + tag + "' did not parse correctly. Check that you have a double colon (::) between each pair of corresponding sentences.")
                v0 = self.encode(examples[i][0])
                v1 = self.encode(examples[i][1])
                #vector = self.encode(examples[i][1]) - self.encode(examples[i][0])
                vector = v1 - v0
                sum_vector += vector
                centroid0 += v0
                centroid1 += v1
                divide_by += 1

            self.canonical_vectors[tag] = sum_vector/divide_by
            self.canonical_centroids[tag] = (centroid0/divide_by, centroid1/divide_by)

        else:
            raise ValueError('Invalid model type for ' + tag + ' in self.canonical_tag_types: ' + model)

    def calculate_canonical_pisa_alphas(self, tag):
        if tag not in self.canonical_tag_types.keys():
            raise ValueError("Tag not found in self.canonical_tag_types")
        model_name = self.canonical_tag_types[tag]

        print("Determining canonical pisa alphas for tag: " + tag)

	vectors_0 = []
	vectors_1 = []
	
        if model_name == 'scholar':
            if tag not in self.canonical_examples.keys():
                raise ValueError("Tag '" + tag + "' not in self.canonical_examples.")

	    examples = self.canonical_examples[tag]
	    print tag
	    print examples
            for i in range(len(examples)):
		print examples[i][0]
		print examples[i][1]
                if self.scholar.exists_in_model(examples[i][1]) and self.scholar.exists_in_model(examples[i][0]):
                    vectors_0.append(self.word_encode(examples[i][0]))
                    vectors_1.append(self.word_encode(examples[i][1]))


        elif model_name == 'penseur':
            if tag not in self.canonical_examples.keys():
                raise ValueError("Tag '" + tag + "' not in self.canonical_examples.")

            for i in range(len(examples)):
                if len(examples[i]) != 2:
                    raise ValueError("Your input file for '" + tag + "' did not parse correctly. Check that you have a double colon (::) between each pair of corresponding sentences.")
                v0.append(self.encode(examples[i][0]))
                v1.append(self.encode(examples[i][1]))


        #For each potential value of alpha...	
    	alpha_values=np.arange(0,1.1,0.1)
	print "\nValues"
	print alpha_values
	alpha_scores = []
	for alpha in alpha_values:
	    #calculate the number of analogies that are answered 'correctly'
	    score = 0
	    for i in range(len(vectors_0)):
	        v0 = vectors_0[i]
	        v1 = vectors_1[i]
	        analogy_point = v1 - v0
		
                #distances = self.get_pisa_scores(vectors_1,analogy_point,self.canonical_vectors[tag],alpha)
                distances = self.get_pisa_scores(self.get_scholar_vectors(),analogy_point,self.canonical_vectors[tag],alpha)
		if np.argmax(distances) == i:
		    score+=1
	
	    alpha_scores.append(score)

	#take the alpha value with the highest score
	print "Scores"
	print alpha_scores
	if np.sum(alpha_scores) == 0:
	    #no value of alpha word for any of the target words
	    #so we'll default to 0.5
	    self.canonical_pisa_alphas[tag] = 0.5
	else:
	    index = np.argmax(alpha_scores)
	    self.canonical_pisa_alphas[tag] = alpha_values[index]
	return self.canonical_pisa_alphas[tag]
	        


    def calculate_canonical_angles(self, tag):
        if tag not in self.canonical_tag_types.keys():
            raise ValueError("Tag not found in self.canonical_tag_types")
        model_name = self.canonical_tag_types[tag]
        print("Creating canonical angles for " + tag)

        if model_name == 'scholar':
            if tag not in self.canonical_examples.keys():
                raise ValueError("Tag '" + tag + "' not in self.canonical_examples.")
            sum_angle = np.zeros([len(self.get_scholar_vectors()[0])]*2)
            #sum_scale = np.zeros([len(self.scholar.model.vectors[0])]*2)
            sum_scale = np.zeros([len(self.get_scholar_vectors()[0])]*2)
            examples = self.canonical_examples[tag]
            divide_by = 0
            for i in range(len(examples)):
                print(examples[i])
                if self.scholar.exists_in_model(examples[i][0]) and self.scholar.exists_in_model(examples[i][1]):
                    #v1 = self.scholar.model[examples[i][0]]
                    #v2 = self.scholar.model[examples[i][1]]
                    v1 = self.word_encode(examples[i][0])
                    v2 = self.word_encode(examples[i][1])
                    hyperangle_matrix, scale_factor = hyperangles.hyperangle(v1,v2)
                    sum_angle+=hyperangle_matrix
                    sum_scale+=scale_factor
                    divide_by += 1

            self.canonical_angles[tag] = sum_angle/divide_by
            self.canonical_scale_factors[tag] = sum_scale/divide_by

        elif model_name == 'penseur':
            if tag not in self.canonical_examples.keys():
                raise ValueError("Tag '" + tag + "' not in self.canonical_examples.")
            vector_length = len(np.squeeze(self.encode('Test')))
            sum_angle = np.zeros([vector_length]*2)
            sum_scale = np.zeros([vector_length]*2)
            examples = self.canonical_examples[tag]
            divide_by = 0
            for i in range(len(examples)):
                print(examples[i])
                v1 = np.squeeze(self.encode(examples[i][0]))
                v2 = np.squeeze(self.encode(examples[i][1]))
                hyperangle_matrix, scale_factor = hyperangles.hyperangle(v1,v2)
                sum_angle+=hyperangle_matrix
                sum_scale+=scale_factor
                divide_by += 1

            self.canonical_angles[tag] = sum_angle/divide_by
            self.canonical_scale_factors[tag] = sum_scale/divide_by

        else:
            raise ValueError('Invalid model type for ' + tag + ' in self.canonical_tag_types: ' + model)


    #def canonical_midpoints(self,tag):
    #        midpoint1 = 0
    #        midpoint2 = 0
    #        ctr = 0
    #        for i in range(len(self.canonical_examples[tag])):
    #            ctr+=1
    #            midpoint1 += self.encode(self.canonical_examples[tag][i][0])
    #            midpoint2 += self.encode(self.canonical_examples[tag][i][1])
    #        return midpoint1,midpoint2


    ########## DISTANCE METRICS ###########

    def distance(self, v1, v2, metric = '',desired_dimensions=[]):
        #this determines what the current default distance metric is
        #for finding closest words, closest sentences, etc.
        if metric=='':
            metric = self.metric

        if metric == 'cosine':
            return self.cosine_distance(v1,v2,desired_dimensions=desired_dimensions)
        elif metric == 'euclidean':
            return self.euclidean(v1,v2,desired_dimensions=desired_dimensions)
        elif metric == 'max_diff':
            return self.max_diff(v1,v2desired_dimensions=desired_dimensions)
        elif metric == 'avg_diff':
            return self.avg_diff(v1,v2,desired_dimensions=desired_dimensions)
        else:
            #assume the value passed was a tag
            #raise ValueError("Invalid value for parameter 'metric': '" + metric + "'")
            #return self.scaled_distance(v1,v2,self.scale_vectors[metric])
            return self.pisa_score(v1,v2,self.canonical_vectors[metric])

    def all_distances(self, v1, v2):
        if isinstance(v1, basestring) or isinstance(v2, basestring):
            raise ValueError("This function takes only vectors as input. Try using sentence_distances() or word_distances() instead.")
        
        print("Cosine distance: " + str(self.cosine_distance(v1,v2)))
        print("Euclidean: " + str(self.euclidean(v1,v2)))
        print("Max diff: " + str(self.max_diff(v1,v2)))
        print("avg diff: " + str(self.avg_diff(v1,v2)))
        print("min diff: " + str(np.min(np.absolute(v2-v1))))
        print("stddev diff: " + str(np.std(np.absolute(v2-v1))))
        if self.use_angles==True:
            print("Calculating angular distance...")
            print("Max 2D projection angle: " + str(self.max_2d_angle(v1,v2)))
            print("Avg 2D projection angle: " + str(self.avg_2d_angle(v1,v2)))
            print("here3")
 
    def cosine_distance(self,v1,v2,desired_dimensions=[]):
        #calculates the cosine distance between two vectors (i.e., dist = 1-cos)
        #identical vectors have a distance of 0
        #orthogonal vectors have a distance of 1
        #a vector and its multiplicative inverse have a distance of 2
        if desired_dimensions==[]:
            if self.desired_dimensions==[]:
                return spatial.distance.cosine(v1,v2)
            else:
                desired_dimensions = self.desired_dimensions

        #create a vector using only the desired dimensions
        v1_prime=np.zeros(len(desired_dimensions))
        v2_prime=np.zeros(len(desired_dimensions))
        ctr=0
        for i in range(len(v1)):
            if i in desired_dimensions:
                if len(v1) > len(desired_dimensions):
                    v1_prime[ctr] = v1[i]
                if len(v2) > len(desired_dimensions):
                    v2_prime[ctr] = v2[i]
                ctr+=1
        if len(v1) > len(desired_dimensions):
            v1_final=v1_prime
        else:
            v1_final=v1
        if len(v2) > len(desired_dimensions):
            v2_final=v2_prime
        else:
            v2_final=v2
        return spatial.distance.cosine(v1_final,v2_final)
        

    def euclidean(self,v1,v2):
        return np.linalg.norm(v2-v1)

    def max_diff(self,v1,v2):
        #returns the maximum distance along a single cardinal dimension
        return np.max(np.absolute((v2-v1)))
    
    def avg_diff(self,v1,v2):
        #returns the average of the distances along each cardinal dimension
        return np.mean(np.absolute((v2-v1)))

    def max_2d_angle(self,v1,v2):
        #returns the maximum angle between the vectors in any 2D angular pojection
        a,s = hyperangles.hyperangle(v1,v2)
        return np.max(np.absolute(a))
    
    def avg_2d_angle(self,v1,v2):
        #returns the avg angle between the vectors in any 2D angular pojection
        a,s = hyperangles.hyperangle(v1,v2)
        return np.mean(np.absolute(a))

    def scaled_distance(self,v1,v2,scale_vector,metric=''):
        v1=np.multiply(v1,scale_vector)
        v2=np.multiply(v2,scale_vector)
        return self.distance(v1,v2,metric)

    def dot_product_score(self, db_word,center_of_search,direction_of_travel):
        pass
        #what if we project all the words onto the vector and use THAT as our distance...

    def get_pisa_scores(self,db_words,center_of_search,direction_of_travel,alpha = -1):

	if alpha == -1:
	    alpha = self.alpha        

        #CALCULATE COSINE SIMILARITIES OF ALL VECTORS TO SOURCE WORD
        #(We normalize center_of_search in case it doesn't have unit length,
        #but we assume db_words are normalized already...)
        base_distances = 1 - np.dot(db_words,center_of_search.T)/np.linalg.norm(center_of_search)
        
        vector_to_source = center_of_search + direction_of_travel
        vectors_to_words = center_of_search - db_words
        supplementary_distances = 1 - np.dot(vectors_to_words,vector_to_source.T)/(np.linalg.norm(vectors_to_words,axis=1) * np.linalg.norm(vector_to_source))

	#IMPORTANT NOTE: I've changed the calculation on the pisa scale to a be a
	#weighted average
        return alpha * (2-supplementary_distances) + (1-alpha) * base_distances

    def pisa_score(self,db_word,center_of_search,direction_of_travel):
        #search in lopsided rings that stretch in the direction of travel
        #(You know, like the leaning tower of Pisa...)

        base_distance = spatial.distance.cosine(db_word,center_of_search)
        vector_to_source = center_of_search + direction_of_travel
        vector_to_word = center_of_search - db_word
        supplementary_distance = spatial.distance.cosine(vector_to_source,vector_to_word)

        scale_factor = self.alpha
        score = scale_factor * (2-supplementary_distance) + base_distance
        
        return score


    def get_unpisa_scores(self,db_words,center_of_search,direction_of_travel):

        #CALCULATE COSINE SIMILARITIES OF ALL VECTORS TO SOURCE WORD
        #(We normalize center_of_search in case it doesn't have unit length,
        #but we assume db_words are normalized already...)
        base_distances = 1 - np.dot(db_words,center_of_search.T)/np.linalg.norm(center_of_search)

        #vector_to_source = center_of_search - direction_of_travel
        vectors_to_words = db_words - center_of_search
        supplementary_distances = 1 - np.dot(vectors_to_words,direction_of_travel.T)/(np.linalg.norm(vectors_to_words,axis=1) * np.linalg.norm(direction_of_travel))

        scale_factor = self.alpha
        return scale_factor * supplementary_distances + base_distances


    def get_superpisa_scores(self,db_words,center_of_search,direction_of_travel,target_point):

        #CALCULATE COSINE SIMILARITIES OF ALL VECTORS TO SOURCE WORD
        #(We normalize center_of_search in case it doesn't have unit length,
        #but we assume db_words are normalized already...)
        base_distances = 1 - np.dot(db_words,center_of_search.T)/np.linalg.norm(center_of_search)

        source_word = center_of_search - direction_of_travel
        vectors_to_words = db_words - center_of_search
        vector_to_target = target_point - source_word
        supplementary_distances = 1 - np.dot(vectors_to_words,vector_to_target.T)/(np.linalg.norm(vectors_to_words,axis=1) * np.linalg.norm(vector_to_target))

        scale_factor = self.alpha
        return scale_factor * supplementary_distances + base_distances



#    def pisa_score2(self,db_word,center_of_search,direction_of_travel):
#        #search in lopsided rings that stretch in the direction of travel
#        #(You know, like the leaning tower of Pisa...)
#
#        #in closest_words(), v1 is the databaseword we're checking, and
#        #v2 is the point we landed on after applying the analogy vector
#
#        base_distance = spatial.distance.cosine(db_word,center_of_search)
#        vector_to_source = center_of_search - direction_of_travel
#        vector_to_word = center_of_search - db_word
#        supplementary_distance = spatial.distance.cosine(vector_to_source,vector_to_word)
#
#        score = (2-supplementary_distance) * base_distance
#        return score
        

        
    def scale_vector(self, samples):
        #calculates the variance along each basis dimension
        #sets the scale factor equal to 2-variance for each dimension
        #returns of a vector of scale factors
        
        if len(samples)==0:
            raise ValueError("Empty sample set cannot be used to generate a vector scale")

        if isinstance(samples[0], basestring):
            
            vectors = []

            if '_' in samples[0] and ' ' not in samples[0]:
                #if the sample contains an underscore but no space,
                #then we assume these are tagged scholar words
                for s in samples:
                    #vectors.append(self.scholar.model[s])
                    vectors.append(self.word_encode(s))
            else:
                #otherwise, we assume they're penseur sentences
                for s in samples:
                    vectors.append(self.encode(s))
        else:
           vectors = samples 

        stacked = np.vstack(vectors)        
        variances = []
        spans = []

        for i in range(len(stacked.T)):
            variance = np.var(stacked.T[i])
            span = np.max(stacked.T[i]) - np.min(stacked.T[i])
            variances.append(variance)
            spans.append(span)
        
        return 1-np.array(variances)
        #return 1-(np.array(variances)/np.max(variances))
        #return 2-np.array(spans)


    def tag_scale(self,tag,index=1):
        #returns a vector indicating the amount each basis dimension
        #should be scaled in distance calculations
        #(scale values are based on variance within the sample)

        model_name = self.model_name(tag)
        examples = self.canonical_examples[tag]
        vectors = []
        
        #print "Tag is " + tag
        #print "Model name is " + model_name        

        #get a set of sample vectors
        if model_name == 'penseur':
            for example in examples:
                vectors.append(self.encode(example[index]))
        elif model_name == 'scholar':
            for example in examples:
                if self.scholar.exists_in_model(example[index]):
                    #vectors.append(self.scholar.model[example[index]])
                    vectors.append(self.word_encode(example[index]))
        else:
            raise ValueError("Unknown model name for tag '" + tag + "'")
        
        if len(vectors) == 0:
            raise ValueError("No canonical examples found for tag '" + tag + "'")

        #find the value by which each basis dimension should be scaled
        return self.scale_vector(vectors)


    ############### distances between vectors ###############

    def closest_word_pair(self, words):
        vectors = []
        keep_words = []
        for w in words:
            if self.scholar.exists_in_model(w):
                #vectors.append(self.scholar.model[w])
                vectors.append(self.word_encode(w))
                keep_words.append(w)
            else:
                print("Word " + w + " not found in model. Omitting...")

        min_cos = 2
        wrds_cos = []
        min_euc = 2
        wrds_euc = []
        min_max_diff = 2
        wrds_max_diff = []
        min_avg_diff = 2
        wrds_avg_diff = []
        for w1,v1 in zip(keep_words,vectors):
            for w2,v2 in zip(keep_words,vectors):
                if np.any(v1 != v2):
                    cos_dist = self.cosine_distance(v1,v2)         
                    euc_dist = self.euclidean(v1,v2)
                    max_diff_dist = self.max_diff(v1,v2)
                    avg_diff_dist = self.avg_diff(v1,v2)

                    if cos_dist < min_cos:
                        min_cos = cos_dist
                        wrds_cos = w1, w2

                    if euc_dist < min_euc:
                        min_euc = euc_dist
                        wrds_euc = w1, w2

                    if max_diff_dist < min_max_diff:
                        min_max_diff = max_diff_dist
                        wrds_max_diff = w1, w2

                    if avg_diff_dist < min_avg_diff:
                        min_avg_diff = avg_diff_dist
                        wrds_avg_diff = w1, w2

        print("Cosine distance:")
        print(wrds_cos)
        print("Euclidean distance:")
        print(wrds_euc)
        print("Maximum vector difference:")
        print(wrds_max_diff)
        print("Average vector difference:")
        print(wrds_avg_diff)

    def closest_sentence_pair(self, sentences):
        vectors = []
        for s in sentences:
            vectors.append(self.encode(s))

        min_cos = 2
        sen_cos = []
        min_euc = 2
        sen_euc = []
        min_max_diff = 2
        sen_max_diff = []
        min_avg_diff = 2
        sen_avg_diff = []
        for s1,v1 in zip(sentences,vectors):
            for s2,v2 in zip(sentences,vectors):
                if np.any(v1 != v2):
                    cos_dist = self.cosine_distance(v1,v2)         
                    euc_dist = self.euclidean(v1,v2)
                    max_diff_dist = self.max_diff(v1,v2)
                    avg_diff_dist = self.avg_diff(v1,v2)

                    if cos_dist < min_cos:
                        min_cos = cos_dist
                        sen_cos = s1, s2

                    if euc_dist < min_euc:
                        min_euc = euc_dist
                        sen_euc = s1, s2

                    if max_diff_dist < min_max_diff:
                        min_max_diff = max_diff_dist
                        sen_max_diff = s1, s2

                    if avg_diff_dist < min_avg_diff:
                        min_avg_diff = avg_diff_dist
                        sen_avg_diff = s1, s2

        print("Cosine distance:")
        print(sen_cos)
        print("Euclidean distance:")
        print(sen_euc)
        print("Maximum vector difference:")
        print(sen_max_diff)
        print("Average vector difference:")
        print(sen_avg_diff)

    def word_tag_distances(self, word, tag):

        #v_source = self.scholar.model[word]        
        v_source = self.word_encode(word)        

        for s in self.canonical_examples[tag]:
            if not self.scholar.exists_in_model(s[0]):
                print("'" + s[0] + "' does not exist in model. Omitting...")
            #elif not self.scholar.exists_in_model(s[1])
            #        print "'" + s[1] + "' does not exist in model. Omitting..."
            else:
                #v = self.scholar.model[s[0]]
                v = self.word_encode(s[0])
                print('\n' + s[0] + ":")
                self.all_distances(v_source, v)

    def closest_pair(self, tag):

        samples = []
        for s in self.canonical_examples[tag]:
            samples.append(s[0])

        if self.canonical_tag_types[tag] == 'penseur':
            self.closest_sentence_pair(samples)
        else:
            self.closest_word_pair(samples)

    def sentence_distances(self, s1, s2):
        v1 = self.encode(s1)
        v2 = self.encode(s2)
        self.all_distances(v1,v2)

    def word_distances(self, w1, w2):
        #v1 = self.scholar.model[w1]
        #v2 = self.scholar.model[w2]
        v1 = self.word_encode(w1)
        v2 = self.word_encode(w2)
        self.all_distances(v1,v2)


    ############ ANALYSIS TOOLS ###############

    def correlations(self, sentences=None, filename='', limit=1000):
        
        if sentences == None:
            with open(filename) as f:
                sentences = f.read().split('\n')

        if len(sentences) == 0:
            raise ValueError("can't create correlations on an empty sentence set.")

        vectors = []
        if '_' in sentences[0] and ' ' not in sentences[0]:
            #assume these are word2vec words
            for s in sentences:
                vectors.append(self.encode(s))
        else:
            #assume these are penseur sentences
            for s in sentences:
                vectors.append(self.word_encode(s))


        #(original correlation code courtesy of http://glowingpython.blogspot.com/2012/10/visualizing-correlation-matrices.html)
        from numpy import corrcoef, sum, log, arange
        from numpy.random import rand
        from pylab import pcolor, show, colorbar, xticks, yticks

        ## generating some uncorrelated data
        #data = rand(10,100) # each row of represents a variable

        data = []
        for v in vectors[:limit]:
            data.append(v)

        data = np.array(data).T.tolist()   

        ## creating correlation between the variables
        ## variable 2 is correlated with all the other variables
        #data[2,:] = sum(data,0)
        ## variable 4 is correlated with variable 8
        #data[4,:] = log(data[8,:])*0.5

        # plotting the correlation matrix
        R = corrcoef(data)
        pcolor(R)
        colorbar()
        yticks(arange(0.5,10.5),range(0,10))
        xticks(arange(0.5,10.5),range(0,10))
        show()
        #savefig('correlations.png')

    def quadrant_counting(self, sentences=None, filename='', limit=1000):

        if sentences == None:
            with open(filename) as f:
                sentences = f.read().split('\n')

        negatives = np.zeros(4800)
        positives = np.zeros(4800)
        sum=0
        for sen in sentences[:limit]:
                v = np.squeeze(s.encode(sen))
                sum += np.linalg.norm(v)
                for i in range(len(v)):
                    if v[i] < 0:
                        negatives[i] = 1
                    if v[i] > 0:
                        positives[i] = 1

        zero_dims=0
        unbalanced_dims=0
        for i in range(len(negatives)):
                if negatives[i] + positives[i] == 0:
                    zero_dims += 1
                if negatives[i] + positives[i] == 1:
                    unbalanced_dims += 1

        print(zero_dims)
        print(unbalanced_dims)


    def squish(self, words, num_dimensions, method='variance'):

        if isinstance(words[0], basestring):
            v_words=[]
            for word in words:
                v_words.append(self.word_encode(word))
        else:
            v_words = words

        if method=='PCA':
            raise ValueError("Sorry, PCA isn't implemented yet.")
        
        if method=='variance':
            #identify the n dimensions with the most variance,
            #and return only those dimensions for each word...
            stacked = np.vstack(v_words);
            variances = np.zeros(len(stacked.T))

            for i in range(len(stacked.T)):
                variances[i] = np.var(stacked.T[i])

            #print variances

        word_dims=np.zeros([num_dimensions,len(words)])
        for i in range(num_dimensions):
            index = np.where(variances==np.max(variances))[0]
            word_dims[i] = stacked.T[index]
            variances[index]=0

        #print word_dims.T

        #normalize to unit length
        for i in range(len(word_dims.T)):
            word_dims.T[i] = word_dims.T[i]/np.linalg.norm(word_dims.T[i])

        return word_dims.T


    ############# EVERYDAY FUNCTIONS ##############

    def model_name(self, tag):
        #returns the model associated with the specified tag
        if tag in self.canonical_tag_types:
            model_name = self.canonical_tag_types[tag]
        else:
           raise ValueError("No entry in self.canonical_tag_types for tag '" + tag + "'")

        return model_name

    def decode(self, vector):
        #WORKS FOR PENSEUR MODEL ONLY
        return self.decode_helper.decode(vector)

    def mode_slice(self, base_encoding, mode):
	    #converts a skipthought vector into a specific representation
	    #created by slicing

	    if mode == 'skipthought':
		pass #do nothing, we want the whole thing
	    elif mode == 'uniskip':
		base_encoding = base_encoding[:2400] #uni-gram (forward) 
	    elif mode == 'biskip':
		base_encoding = base_encoding[2400:] #bigram (forward+backward)
	    elif mode == 'reverse':
		base_encoding = base_encoding[3600:] #bigram (backward only)
	    elif mode == 'forward':
		base_encoding = base_encoding[2400:3600] #bigram (forward only)
	    elif mode == 'tforward':
		base_encoding = base_encoding[:1200] 
	    elif mode == 'treverse':
		base_encoding = base_encoding[1200:2400] 
	    elif mode == 'mini':
		base_encoding = base_encoding[:100]
	    else:
		raise ValueError('Unrecognized encoding mode: ' + mode)

	    return base_encoding


    def encode(self, sentence, mode='skipthought'):
        if self.universal == True:
            if mode == 'skipthought':
                mode = 'universal'
        if mode == 'universal':
            sentence_embedding = self.session.run(self.embed([sentence]))

            return np.squeeze(np.array(sentence_embedding))

        #WORKS FOR PENSEUR MODEL ONLY
	sentence = sentence.strip()
	if self.cached_encoding_mode == mode and sentence in self.cached_encodings.keys():
	    return self.cached_encodings[sentence]
	else:

	    if mode == 'average_of_words':
		words = sentence.split(' ')
		word_sum = np.zeros(100) #100-dimensional word vectors
		count = 0
		for w in words:
		    #w = w.strip(['.',',',';',':','!','?','-','`','\'','"'])
		    w = w.strip('.,;:!?-`"()[]{}')
		    tagged = self.tag(w)
		    if self.scholar.exists_in_model(tagged):
		   	v_w = self.word_encode(self.tag(w))
		    	word_sum += v_w
			count += 1

		base_encoding = word_sum/count

	    else:
		#mode = 'skipthought' or one of several 'sliced' skipthoughts
	        base_encoding = np.squeeze(self.penseur.get_vector(sentence))
	        base_encoding = self.mode_slice(base_encoding, mode)

	    self.cached_encodings[sentence] = base_encoding
	    self.cached_encoding_mode = mode
            return self.cached_encodings[sentence]
 
    def word_encode(self, word, desired_dimensions=[]):
        #WORKS FOR SCHOLAR MODEL ONLY
        if desired_dimensions == []:
            if self.desired_dimensions==[]:
	       if len(word.split(' ')) > 1:
	  	   #it's a mult-word phrase, we use just the last word...
                   return self.scholar.model[word.split(' ')[-1]]
	       else:
                   return self.scholar.model[word]
            else:
                desired_dimensions=self.desired_dimensions
        
        v_word = self.scholar.model[word]
        v_return=np.zeros(len(desired_dimensions))
        ctr=0
        for i in range(len(v_word)):
            if i in desired_dimensions:
                v_return[ctr] = v_word[i]
                ctr+=1
        return v_return

        #an experiment:
        #what if we use the eigenvectors of the entire word corpus to
        #find the most significant 10 dimensions, and then use those
        #ten dimensions for all subsequent calculations...

    def vectorize(self, sentences):
        vectors = []
        for s in sentences:
            #conformity sacrificed for efficiency...
            #vectors.append(self.encode(self.penseur.get_vector(sentence)))
            vectors.append(self.encode(s))
        return vectors

    def strip_tags(self, word_list):
        untagged=[]
        for w in word_list:
            if '_' in w:
                untagged.append(w[:w.index('_')])
            else:
                untagged.append(w)
        return untagged

    def add_tags(self, word_list):
        tagged = []
        for w in word_list:
            tagged.append(w + '_' + self.scholar.get_most_common_tag(w.lower()))
        return tagged
    
    def untag(self, word):
        return word[:word.index('_')]

    def tag(self, word):
	try:
            return word + '_' + self.scholar.get_most_common_tag(word.lower())
	except:
	    return word + '_??'
    
    def strip_sentence_tags(self, sentence_list):
        untagged=[]
        for s in sentence_list:
            new_sentence = []
            for w in s.split(' '):
              if '_' in w:
                  new_sentence.append(w[:w.index('_')])
              else:
                  new_sentence.append(w)
            untagged.append(' '.join(new_sentence))
        return untagged

    def add_sentence_tags(self, sentence_list):
        tagged = []
        for s in sentence_list:
            new_sentence = []
            for w in s.split(' '):
                new_sentence.append(w + '_' + self.scholar.get_most_common_tag(w))
            tagged.append(' '.join(new_sentence))
        return tagged

    def get_scholar_vectors(self):
        return self.scholar.model.vectors
    
    def get_penseur_vectors(self):
        return self.penseur.vectors

    def sentence_average(self, sentence1, sentence2, normalize=True, stddev=None):
        if isinstance(sentence1, basestring):
            v1 = self.encode(sentence1)
        else:
            v1 = sentence1

        if isinstance(sentence2, basestring):
            v2 = self.encode(sentence2)
        else:
            v2 = sentence2

        v_avg = (v1 + v2) / 2.0

        print("Distance between source and target sentences")
        print(self.all_distanceS(v1,v2))

        print("\nNorm of averaged vector: " + str(np.linalg.norm(v_avg)))
        if normalize==True:
            return self.jitter(v_avg/np.linalg.norm(v_avg), stddev)
        else:
            return self.jitter(v_avg, stddev)


    def word_average(self, word1, word2):
        if isinstance(word1, basestring):
            #w1 = self.scholar.model[word1]
            w1 = self.word_encode(word1)
        else:
            w1 = word1

        if isinstance(word2, basestring):
            #w2 = self.scholar.model[word2]
            w2 = self.word_encode(word2)
        else:
            w2 = word2

        w_avg = (w1 + w2)/2.0

        return self.closest_words(w_avg)

    def word_jitter(self, word, stddev=None, num_results=10):
        if isinstance(word, basestring):
            #w = self.scholar.model[word]
            w = self.word_encode(word)
        else:
            w = word

        jitter_results=[]
        nearest_match = self.closest_words(word, 1)[0]
        jitter_results.append(nearest_match)
        for i in range(num_results):
            if stddev == None:
                w_prime = w + np.random.normal(0,self.stddev,len(w))
            else:
                w_prime = w + np.random.normal(0,stddev,len(w))
            nearest_match = self.closest_words(w_prime, 1)[0]
            jitter_results.append(nearest_match)
            
        return jitter_results


    def jitter(self, sentence, stddev=None, num_results=10):
        #adds random amounts to the vector, as a way of sampling
        #the space near a given sentence
        if isinstance(sentence, basestring):
            v = self.encode(sentence)
        else:
            v = sentence

        jitter_results = []
        jitter_results.append(self.decode(v.astype(np.float32)))
        for i in range(num_results):
            if stddev == None:
                v_prime = v + np.random.normal(0,self.stddev,len(v))
            else:
                v_prime = v + np.random.normal(0,stddev,len(v))
            jitter_results.append(self.decode(v_prime.astype(np.float32)))
        
        return jitter_results

    def closest_sentences(self, vector, num_matches=5, return_distance=False, metric='',desired_dimensions=[]):
        
        if self.penseur.vectors is None:
            print("No pre-encoded sentences found.")
            return None

        if isinstance(vector, basestring):
            #if a string was passed in, convert it to
            #a vector...
            vector = self.encode(vector)

        distances = []
        for v in self.penseur.vectors:
            #distances.append(abs(spatial.distance.cosine(vector, v)))
            if metric=='':
                distances.append(self.distance(vector, v,desired_dimensions=desired_dimensions))
            else:
                distances.append(self.distance(vector, v, metric))

        #print distances[:10]

        found_sentences = []
        found_distance = []
        for count in range(0, num_matches):
            min_dist = min(distances)
            index = distances.index(min_dist)
            found_sentences.append(self.penseur.get_sentence(self.penseur.vectors[index]))
            found_distance.append(min_dist)
            distances[index] = 100

        if return_distance == True:
            return found_sentences, found_distance
        else:
            return found_sentences

    def proximity_sort(self, v_target, vectors, strings, return_distance=False, cutoff=2, use_pisa = False, pisa_vector=None):
        #NOTE: This function optimizes for speed, and hence assumes
        #that all the comparison vectors are already normalized
        if use_pisa == True:
	    if pisa_vector is None:
		raise ValueError("HEY!!! -- you said use_pisa=True, but you didn't specify a pisa_vector.")
            distances = self.get_pisa_scores(vectors,v_target,pisa_vector)
        else:
            distances = 1 - np.dot(vectors,v_target.T)/np.linalg.norm(v_target)
            #distances = np.dot(vectors,v_target.T)/(np.linalg.norm(v_target) * np.linalg.norm(vectors[0]))

        found_words = []
        found_distance = []
        for count in range(0, len(vectors)):
            min_dist = np.nanmin(distances)
            if min_dist <= cutoff:
                index = np.where(distances==min_dist)[0][0]
                found_words.append(strings[index])
                found_distance.append(min_dist)
                distances[index] = 1000


        if return_distance==True:
            return found_words,found_distance
        else:
            return found_words

    def projection_sort(self, vectors, strings, v_projection, cutoff=2, return_distance = False):
        projected_vectors = np.dot(v_projection, np.squeeze(np.array(vectors))).T
        
        zipped = zip(projected_vectors, strings)
        zipped.sort()
        sorted_vectors = [projected_vectors for (projected_vectors, strings) in zipped]
        sorted_strings = [strings for (projected_vectors, strings) in zipped]

        if return_distance == True:
            return sorted_strings, sorted_vectors
        else:
            return sorted_strings


    def closest_pisa_words(self,word,num_matches=15,return_distance=False,metric='',analogy_vector=None,scale_vector=None, desired_dimensions=[]):
        if metric != '' and scale_vector != None:
            print("WARNING: You've specified both an alternate distance metric and a scale_vector. Only the metric will be used.")
        
        if metric != '' and analogy_vector != None:
            print("WARNING: You've specified both a distance metric and an analogy_vector. Only the analogy_vector will be used.")
        
        
        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word,desired_dimensions)
        elif isinstance(word, (np.ndarray, np.generic) ):
            v_word = word
        else:
            raise ValueError("Unknown parameter type in closest_words()")

        if not analogy_vector is None:
            distances = self.get_pisa_scores(self.get_scholar_vectors(),v_word,analogy_vector)
        else:
            distances = self.get_pisa_scores(self.get_scholar_vectors(),v_word,self.canonical_vectors[metric])
        distances=distances.tolist()        

        found_words = []
        found_distance = []
        for count in range(0, num_matches):
            min_dist = np.nanmin(distances)
            index = np.where(distances==min_dist)[0][0]
            found_words.append(self.scholar.model.vocab[index])
            found_distance.append(min_dist)
            distances[index] = 1000


        if return_distance==True:
            return found_words, found_distance
        else:
            return found_words


    def closest_pisa_sentences(self,sentence,num_matches=15,return_distance=False,metric='',analogy_vector=None,scale_vector=None, desired_dimensions=[]):
        if metric != '' and scale_vector != None:
            print("WARNING: You've specified both an alternate distance metric and a scale_vector. Only the metric will be used.")
        
        if metric != '' and analogy_vector != None:
            print("WARNING: You've specified both a distance metric and an analog_vector. Only the analogy_vector will be used.")
        
        
        #if the sentence is a string, convert it to a vector
        if isinstance(sentence, basestring):
            v_sentence = self.encode(sentence)
        else:
            v_sentence = sentence

        if not analogy_vector is None:
            distances = self.get_pisa_scores(self.get_penseur_vectors(),v_sentence,analogy_vector)
        else:
            distances = self.get_pisa_scores(self.get_penseur_vectors(),v_sentence,self.canonical_vectors[metric])
        distances=distances.tolist()        

        found_sentences = []
        found_distance = []
        for count in range(0, num_matches):
            min_dist = np.nanmin(distances)
            index = np.where(distances==min_dist)[0][0]
            found_sentences.append(self.penseur.get_sentence(self.get_penseur_vectors()[index]))
            found_distance.append(min_dist)
            distances[index] = 1000


        if return_distance==True:
            return found_sentences, found_distance
        else:
            return found_sentences



    def closest_unpisa_words(self,word,num_matches=15,return_distance=False,metric='',analogy_vector=None,scale_vector=None, desired_dimensions=[]):
        if metric != '' and scale_vector != None:
            print("WARNING: You've specified both an alternate distance metric and a scale_vector. Only the metric will be used.")
        
        if metric != '' and analogy_vector != None:
            print("WARNING: You've specified both a distance metric and an analog_vector. Only the analogy_vector will be used.")
        
        
        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word,desired_dimensions)
        elif isinstance(word, (np.ndarray, np.generic) ):
            v_word = word
        else:
            raise ValueError("Unknown parameter type in closest_words()")

        if not analogy_vector is None:
            distances = self.get_unpisa_scores(self.get_scholar_vectors(),v_word,analogy_vector)
        else:
            distances = self.get_unpisa_scores(self.get_scholar_vectors(),v_word,self.canonical_vectors[metric])
        distances=distances.tolist()        

        found_words = []
        found_distance = []
        for count in range(0, num_matches):
            min_dist = np.nanmin(distances)
            index = np.where(distances==min_dist)[0][0]
            found_words.append(self.scholar.model.vocab[index])
            found_distance.append(min_dist)
            distances[index] = 1000


        if return_distance==True:
            return found_words, found_distance
        else:
            return found_words


    def closest_superpisa_words(self,word,num_matches=15,return_distance=False,metric='',analogy_vector=None,target_point=None,scale_vector=None, desired_dimensions=[]):
        if metric != '' and scale_vector != None:
            print("WARNING: You've specified both an alternate distance metric and a scale_vector. Only the metric will be used.")
        
        if metric != '' and analogy_vector != None:
            print("WARNING: You've specified both a distance metric and an analog_vector. Only the analogy_vector will be used.")
        
        
        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word,desired_dimensions)
        elif isinstance(word, (np.ndarray, np.generic) ):
            v_word = word
        else:
            raise ValueError("Unknown parameter type in closest_words()")

        if not analogy_vector is None:
            distances = self.get_superpisa_scores(self.get_scholar_vectors(),v_word,analogy_vector,target_point)
        else:
            distances = self.get_superpisa_scores(self.get_scholar_vectors(),v_word,self.canonical_vectors[metric],self.canonical_target_points[metric])
        distances=distances.tolist()        

        found_words = []
        found_distance = []
        for count in range(0, num_matches):
            min_dist = np.nanmin(distances)
            index = np.where(distances==min_dist)[0][0]
            found_words.append(self.scholar.model.vocab[index])
            found_distance.append(min_dist)
            distances[index] = 1000


        if return_distance==True:
            return found_words, found_distance
        else:
            return found_words

    def closest_context_words(self, word, num_matches=15, return_distance=False, context=None):

        if isinstance(word, basestring):
	    v_word = self.word_encode(word)
	else:
	    v_word = word

	vectors = self.get_scholar_vectors()
	if context is not None:
            if isinstance(context, basestring):
	        v_context = self.word_encode(context)
	    else:
		v_context = context
	    vectors=np.multiply(vectors, v_context)

        distances=[]
        for v in vectors:
            distances.append(spatial.distance.euclidean(v, v_word))
        
	found_words = []
        found_distance = []
        for count in range(0, num_matches):
            min_dist = np.nanmin(distances)
            index = np.where(distances==min_dist)[0][0]
            found_words.append(self.scholar.model.vocab[index])
            found_distance.append(min_dist)
            distances[index] = 1000

        if return_distance==True:
            return found_words,found_distance
        else:
            return found_words



    def closest_words_fast(self, v_word, num_matches=15, return_distance=False,context=''):
	vectors = self.get_scholar_vectors()
	if context != '':
	    raise ValueError("closest_words_fast needs debugging when a context is used. It uses cosine similarity, which is invalid when the incoming vectors do not have a norm of one. (Which is the case when everything has been scaled based on a context)")
            if isinstance(context, basestring):
	        v_context = self.word_encode(context)
	    else:
		v_context = context
	    vectors=1 - np.multiply(vectors, v_context)

        distances = 1 - np.dot(vectors,v_word.T)/np.linalg.norm(v_word)
  
        found_words = []
        found_distance = []
        for count in range(0, num_matches):
            min_dist = np.nanmin(distances)
            index = np.where(distances==min_dist)[0][0]
            found_words.append(self.scholar.model.vocab[index])
            found_distance.append(min_dist)
            distances[index] = 1000

        if return_distance==True:
            return found_words,found_distance
        else:
            return found_words

    def closest_words(self, word, num_matches=15, return_distance=False, metric='', scale_vector=None, desired_dimensions=[]):
        if metric != '' and scale_vector != None:
            print("WARNING: You've specified both an alternate distance metric and a scale_vector. Only the metric will be used.")

        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word,desired_dimensions)
        elif isinstance(word, (np.ndarray, np.generic) ):
            v_word = word
        else:
            raise ValueError("Unknown parameter type in closest_words()")

        distances=[]
        for v in self.get_scholar_vectors():
            #distances.append(abs(spatial.distance.cosine(v, v_word)))
            if metric == '':
                if scale_vector == None:
                   distances.append(self.distance(v, v_word, desired_dimensions=desired_dimensions))
                else:
                   distances.append(self.scaled_distance(v, v_word,scale_vector,desired_dimensions=desired_dimensions))
                
            else:
                distances.append(self.distance(v, v_word, metric=metric, desired_dimensions=desired_dimensions))

        found_words = []
        found_distance = []
        for count in range(0, num_matches):
            min_dist = min(distances)
            index = distances.index(min_dist)
            found_words.append(self.scholar.model.vocab[index])
            found_distance.append(min_dist)
            distances[index]=1000

        if return_distance==True:
            return found_words, found_distance
        else:
            return found_words

    def canonical_projection_old(self,source_word,tag,num_matches=15):

        words = self.closest_words(source_word,num_matches=num_matches)
        projections=[]
        for w in words:
           v_w = self.word_encode(w)
           projections.append(v_w.dot(self.canonical_vectors[tag]))

        print(words)
        print(projections)
        return [w for (p,w) in sorted(zip(projections,words))]
           
    def canonical_projection(self,source_vector,tag):
        return source_vector.dot(self.canonical_vectors[tag])
    
    def canonical_multiprojection(self,source_vectors,tag):
        return source_vectors.dot(self.canonical_vectors[tag])


    def sentences_within_radius(self, start_sentence, search_radius, sentence_list, return_distance=False, metric='', scale_vector=None):
        if metric != '' and scale_vector != None:
            print("WARNING: You've specified both an alternate distance metric and a scale_vector. Only the metric will be used.")
        
        if len(sentence_list) == 0:
            raise ValueError("Cannot find matches within an empty sentence list.")

        if isinstance(start_sentence, basestring):
            v_start = self.encode(start_sentence)
        else:
            v_start = start_sentence

        distances=[]
        found_sentences=[]
        
        #if the words are strings, convert them to vectors
        for sentence in sentence_list:
            if isinstance(sentence, basestring):
                v_sentence = self.encode(sentence)
            else:
                v_sentence = sentence

            if metric == '':
                if scale_vector == None:
                   distance = self.distance(v_start, v_sentence)
                else:
                   distance=self.scaled_distance(v_start, v_sentence,scale_vector)
                
            else:
                distance = self.distance(v_start, v_sentence, metric=metric)
        
            if distance < search_radius:
                found_sentences.append(sentence)
                distances.append(distance)

        if return_distance==True:
            return found_sentences, distances
        else:
            return found_sentences

    def words_within_radius(self, start_word, search_radius, word_list, return_distance=False, metric='', scale_vector=None):
        if metric != '' and scale_vector != None:
            print("WARNING: You've specified both an alternate distance metric and a scale_vector. Only the metric will be used.")
        
        if len(word_list) == 0:
            raise ValueError("Cannot find matches within an empty word list.")

        if isinstance(start_word, basestring):
            v_start = self.word_encode(start_word)
        else:
            v_start = start_word

        distances=[]
        found_words=[]
        
        #if the words are strings, convert them to vectors
        for word in word_list:
            if isinstance(word, basestring):
                v_word = self.word_encode(word)
            else:
                v_word = word

            if metric == '':
                if scale_vector == None:
                   distance = self.distance(v_start, v_word)
                else:
                   distance=self.scaled_distance(v_start, v_word,scale_vector)
                
            else:
                distance = self.distance(v_start, v_word, metric=metric)
        
            if distance < search_radius:
                found_words.append(word)
                distances.append(distance)

        if return_distance==True:
            return found_words, distances
        else:
            return found_words

    def get_angular_word_relation(self, word, tag, num_matches=5, return_distance=True, invert=False,metric=''):
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word)
        else:
            v_word = word

        if tag not in self.canonical_angles.keys():
            raise ValueError("Tag " + tag + " not in canonical_angles.keys()")

        if invert==True:
            v_result = hyperangles.unit_rotation(self.canonical_angles[tag], self.canonical_scale_factors[tag], v_word)
        else:
            v_result = hyperangles.unit_rotation(self.canonical_angles[tag], self.canonical_scale_factors[tag], v_word)

        print(self.closest_words(v_result, num_matches=num_matches, return_distance=return_distance, metric=metric))
        return v_result
        

    def get_angular_sentence_relation(self, sentence, tag, num_matches=5, return_distance=True, invert=False, metric='', stddev=None):
        if isinstance(sentence, basestring):
            v_sentence = self.encode(sentence)
        else:
            v_sentence = sentence

        if tag not in self.canonical_angles.keys():
            raise ValueError("Tag " + tag + " not in canonical_angles.keys()")

        if invert==True:
            v_result = hyperangles.unit_rotation(self.canonical_angles[tag], self.canonical_scale_factors[tag], v_sentence)
        else:
            v_result = hyperangles.unit_rotation(self.canonical_angles[tag], self.canonical_scale_factors[tag], v_sentence)
        
        print("\n\nThe norm of the returned sentence is " + str(np.linalg.norm(v_sentence)))
        print("The distance from the source sentence is...")
        print(self.all_distances(v_sentence, v_result))
        print("\nUn-normalized jitter...")
        print(self.jitter(v_result, stddev=stddev))
        print("Normalised jitter...")
        print(self.jitter(v_result/np.linalg.norm(v_result), stddev=stddev))
        print(self.closest_sentences(v_result))
        return v_result

    def get_angular_relation(self, input_str, tag, num_matches=5, return_distance=True, invert=False, metric='', stddev=None):
        print('\n')
        if tag not in self.canonical_tag_types.keys():
            raise ValueError("Invalid tag: '" + tag + "'")

        if self.canonical_tag_types[tag] == 'penseur':
            return self.get_angular_sentence_relation(input_str, tag, num_matches, return_distance=True, invert=invert,metric=metric, stddev=stddev)
        elif self.canonical_tag_types[tag] == 'scholar':
            return self.get_angular_word_relation(input_str, tag, num_matches, return_distance=True, invert=invert,metric=metric)
        else:
            raise ValueError("Tag '" + tag + "' not listed in self.canonical_tag_types")

    def get_angular_affordance(self, word, num_matches=5):
        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word)
        elif isinstance(word, (np.ndarray, np.generic) ):
            v_word = word
        else:
            print("ERROR: unknown data type for Strategist.get_affordance():")
            print(type(effect))
            sys.exit()
        
        v_result = hyperangles.unit_rotation(self.canonical_angles['affordance'], self.canonical_scale_factors['affordance'], v_word)        

        print(self.closest_words(v_result, num_matches, return_distance=True))
        return v_result

    def apply_nearest_word_relation(self, word, tag, num_matches=5, return_distance=True, invert=False, metric=''):
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word)
        else:
            v_word = word

        min_distance = 1000
        canonical_vector = np.zeros(len(v_word))
        nearest_example = ''
        v_nearest_example = None
        for example in self.canonical_examples[tag]:
            if self.scholar.exists_in_model(example[0]) and self.scholar.exists_in_model(example[1]):

                if invert == True:
                    #v_example = self.scholar.model[example[1]]
                    v_example = self.word_encode(example[1])
                else:
                    #v_example = self.scholar.model[example[0]]
                    v_example = self.word_encode(example[0])

                if metric=='':
                    distance = self.distance(v_word, v_example)
                else:
                    distance = self.distance(v_word, v_example, metric)

                if distance < min_distance:
                    min_distance = distance
                    v_nearest_example = v_example
                    canonical_vector = np.squeeze(self.word_encode(example[1]) - self.word_encode(example[0]))
                    if invert == True:
                        canonical_vector = -1 * canonical_vector
                        nearest_example = example[1]
                    else:
                        nearest_example = example[0]
            else:
                if not self.scholar.exists_in_model(example[0]):
                    print("'" + example[0] + "' not in model. Omitting...")
                if not self.scholar.exists_in_model(example[1]):
                    print("'" + example[1] + "' not in model. Omitting...")
                
            
        print("\nNearest example: " + nearest_example)
        print(self.all_distances(v_word, v_nearest_example))

        v_result = v_word + canonical_vector
        print("Canonical vector length (Euclidean): " + str(np.linalg.norm(canonical_vector)))
        print(self.closest_words(v_result, return_distance=return_distance))
        return v_result



    def apply_nearest_sentence_relation(self, sentence, tag, num_matches=5, return_distance=True, invert=False, metric='', stddev=None):
        print('\n\n')

        if isinstance(sentence, basestring):
            v_sentence = self.encode(sentence)
        else:
            v_sentence = sentence

        min_distance = 1000
        canonical_vector = np.zeros(len(v_sentence))
        nearest_example = ''
        v_nearest_example = None
        for example in self.canonical_examples[tag]:
            if invert == True:
                v_example = self.encode(example[1])
            else:
                v_example = self.encode(example[0])

            if metric=='':
                distance = self.distance(v_sentence, v_example)
            else:
                distance = self.distance(v_sentence, v_example, metric)

            if distance < min_distance:
                min_distance = distance
                v_nearest_example = v_example
                canonical_vector = self.encode(example[1]) - self.encode(example[0])
                if invert == True:
                    canonical_vector = -1 * canonical_vector
                    nearest_example = example[1]
                else:
                    nearest_example = example[0]

        v_result = v_sentence + canonical_vector
            
        print("\nNearest example: " + nearest_example)
        print(self.all_distances(v_sentence, v_nearest_example))

        print("Canonical vector length (Euclidean): " + str(np.linalg.norm(canonical_vector)))

        print("\nThe norm of the returned sentence is " + str(np.linalg.norm(v_result)))
        print("The distance from the source sentence is:")
        print(self.all_distances(v_sentence, v_result))
        print("Un-normalized jitter...")
        print(self.jitter(v_result, stddev=stddev))
        print("Normalised jitter...")
        print(self.jitter(v_result/np.linalg.norm(v_result), stddev=stddev))
        print(self.closest_sentences(v_result))
        return v_result


    def apply_nearest_relation(self, input_str, tag, num_matches=5, return_distance=True, invert=False, metric='', stddev=None):
        print('\n')
        if tag not in self.canonical_tag_types.keys():
            raise ValueError("Invalid tag: '" + tag + "'")

        if self.canonical_tag_types[tag] == 'penseur':
            return self.apply_nearest_sentence_relation(input_str, tag, num_matches, return_distance=True, invert=invert, metric=metric, stddev=stddev)
        elif self.canonical_tag_types[tag] == 'scholar':
            return self.apply_nearest_word_relation(input_str, tag, num_matches, return_distance=True, invert=invert, metric=metric)
        else:
            raise ValueError("Tag '" + tag + "' not listed in self.canonical_tag_types")

    def get_sentence_relation(self, sentence, tag, num_matches=15,return_distance=False,invert=False, metric='', stddev=None, exclude_ratio=None,canonical_scale_factor=1):
        #if the sentences is a string, convert it to a vector
        if isinstance(sentence, basestring):
            v_sentence = self.encode(sentence)
        else:
            v_sentence = sentence

        if invert==True:
            v_result = v_sentence - canonical_scale_factor*self.canonical_vectors[tag]
        else:
            v_result = v_sentence + canonical_scale_factor*self.canonical_vectors[tag]
        print("The norm of the returned sentence is " + str(np.linalg.norm(v_result)))
        print("The distance from the source sentence is...")
        print(self.all_distances(v_sentence, v_result))
        print("\nUn-normalized jitter...")
        print(self.jitter(v_result, stddev=stddev))
        print("Normalised jitter...")
        print(self.jitter(v_result/np.linalg.norm(v_result),stddev=stddev))
        found_sentences= self.closest_sentences(v_result, num_matches=num_matches,return_distance=return_distance,metric=metric)
        print("\nFOUND SENTENCES")
        print(found_sentences)
        print(self.closest_sentences(v_result, num_matches=num_matches, return_distance=return_distance,metric=tag))

        #exclude phrases that are 'close' to the starting location
        if exclude_ratio != None:
           print("excluded sentences:")
           exclude_vector = v_sentence
           exclude_radius = exclude_ratio*canonical_scale_factor*np.linalg.norm(self.canonical_vectors[tag])
           excluded_sentences = self.sentences_within_radius(exclude_vector, exclude_radius, found_sentences, return_distance=False, metric=metric)
           print(self.sentences_within_radius(exclude_vector, exclude_radius, found_sentences, return_distance=return_distance, metric=metric))

           final_sentences=[]
           for sentence in found_sentences:
                if sentence not in excluded_sentences:
                    final_sentences.append(sentence)

           print("final sentences:")
           print(final_sentences)

        return v_result
        


    def get_word_relation(self, word, tag, num_matches=5, return_distance=True, invert=False, metric='', exclude_ratio=None,canonical_scale_factor=1):
        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word)
        elif isinstance(word, (np.ndarray, np.generic) ):
            v_word = word
        else:
            print("ERROR: unknown data type for Strategist.get_affordance():")
            print(type(effect))
            sys.exit()

        if invert==True:
            v_result = v_word - canonical_scale_factor*self.canonical_vectors[tag]
        else:
            v_result = v_word + canonical_scale_factor*self.canonical_vectors[tag]
        #print self.word_jitter(v_result)

        #find words that match
        #found_words = self.closest_words(v_result, num_matches=num_matches, return_distance=False, metric=metric)
        found_words = self.closest_words_fast(v_result, num_matches=num_matches)
        #print "CANONICAL VECTOR LENGTH:" + str(np.linalg.norm(self.canonical_vectors[tag]))
        #print found_words
        #print self.closest_words(v_result, num_matches=num_matches, return_distance=return_distance, metric=tag)
        #print self.closest_pisa_words(v_result, num_matches=num_matches, analogy_vector=self.canonical_vectors[tag])

        #exclude words that are 'close' to the starting location
        if exclude_ratio != None:
           print("excluded words:")
           exclude_vector = v_word
           exclude_radius = exclude_ratio*canonical_scale_factor*np.linalg.norm(self.canonical_vectors[tag])
           excluded_words = self.words_within_radius(exclude_vector, exclude_radius, found_words, return_distance=False, metric=metric)
           print(self.words_within_radius(exclude_vector, exclude_radius, found_words, return_distance=return_distance, metric=metric))

           final_words=[]
           for word in found_words:
                if word not in excluded_words:
                    final_words.append(word)

           print("final words:")
           print(final_words)

        return v_result

    def get_relation(self, input_str, tag, num_matches=15, invert=False, return_distance=False, metric='',stddev=None, exclude_ratio=None,canonical_scale_factor=1):
        if tag not in self.canonical_tag_types.keys():
            raise ValueError("Invalid tag: '" + tag + "'")

        if self.canonical_tag_types[tag] == 'penseur':
            return self.get_sentence_relation(input_str, tag, num_matches, return_distance=return_distance, invert=invert, metric=metric,stddev=stddev, exclude_ratio=exclude_ratio,canonical_scale_factor=canonical_scale_factor)
        elif self.canonical_tag_types[tag] == 'scholar':
            return self.get_word_relation(input_str, tag, num_matches, return_distance=return_distance, invert=invert, metric=metric, exclude_ratio=exclude_ratio,canonical_scale_factor=canonical_scale_factor)
        else:
            raise ValueError("Tag '" + tag + "' not listed in self.canonical_tag_types")


    def centroid_relation(self, input_str, tag='drone_navigation', return_distance=False, all_distances=False):
        #FOR DRONE NAVIGATION ONLY - SPECIALIZED TO THAT TASK
        if tag not in self.canonical_tag_types.keys():
            raise ValueError("Invalid tag: '" + tag + "'")

        if self.canonical_tag_types[tag] == 'penseur':
            start_vector = self.encode(input_str)
            analogy_point = start_vector + self.canonical_vectors[tag]
            distances = []
            for v in self.drone_vectors:
                distances.append(spatial.distance.cosine(v, analogy_point))
            index = distances.index(min(distances))
            if return_distance == True:
                if all_distances == True:
                    print zip(self.drone_examples,distances)
                return self.drone_examples[index], distances[index]
            else:
                return self.drone_examples[index]

        else:
            raise ValueError("Tag '" + tag + "' not listed in self.canonical_tag_types for Penseur")


    def nearest_example_relation(self, input_str, tag='drone_navigation', return_distance=False, all_distances=False):
        #FOR DRONE NAVIGATION ONLY - SPECIALIZED TO THAT TASK
        if tag not in self.canonical_tag_types.keys():
            raise ValueError("Invalid tag: '" + tag + "'")

        if self.canonical_tag_types[tag] == 'penseur':
            start_vector = self.encode(input_str)
    
            #find the closest analogy example
            distances = []
            for example in self.canonical_examples[tag]:
                v = self.encode(example[0])
                distances.append(spatial.distance.cosine(v, start_vector))
            index = distances.index(min(distances))
            print 'nearest example is ' + self.canonical_examples[tag][index][0]
            analogy_vector = self.encode(self.canonical_examples[tag][index][1]) - self.encode(self.canonical_examples[tag][index][0])

            analogy_point = start_vector + analogy_vector
            distances = []
            for v in self.drone_vectors:
                distances.append(spatial.distance.cosine(v, analogy_point))
            index = distances.index(min(distances))
            if return_distance == True:
                if all_distances == True:
                    print zip(self.drone_examples,distances)
                return self.drone_examples[index], distances[index]
            else:
                return self.drone_examples[index]

        else:
            raise ValueError("Tag '" + tag + "' not listed in self.canonical_tag_types for Penseur")


    def KNNclassify(self, sentence, tag='KEG',k=5):
	#Looks at the five nearest neighbors (from the positive canonical examples)

	v_sen = self.encode(sentence)

	#in order to determine which category best applies
	if tag == 'KEG':
	    class_examples = {}
	    class_counts={}
	    for key in self.canonical_examples.keys():
		if 'KEG' in key:
		    class_counts[key] = 0
		    for example in self.canonical_examples[key]:
			class_examples[example[1]] = key

	    #print 'class examples are:'
	    #print class_examples
	    distances=[]
	    for sen in class_examples.keys(): #for each exemplar
	    	v = self.encode(sen)
	    	distances.append(self.cosine_distance(v,v_sen))
	    keys = [key for key,dist in sorted(zip(class_examples.keys(),dist))]
	    print 'nearest sentences are:'
	    print keys
	    for i in range(k):
		class_counts[class_examples[keys[i]]] += 1
	    print 'final counts'
	    print class_counts

	    maxcount = 0
	    output_class = ''
	    for k in class_counts.keys():
		if class_counts[k] > maxcount:
		    max_count = class_counts[k]
		    output_class = k
	    return output_class

	#if tag == 'simon_invoked':

	raise ValueError('Unrecognized KNN tag: ' + tag)

    def categorize(self, sentence, tag, mode='skipthought'):
        #returns true if the sentence is an example of the
        #rightmost category in the input file corresponding
        #to tag

        if isinstance(sentence, basestring):
            v_sentence = self.encode(sentence,mode)
        else:
            v_sentence = sentence

        midpoint0 = self.mode_slice(self.canonical_centroids[tag][0], mode)
        midpoint1 = self.mode_slice(self.canonical_centroids[tag][1], mode)
        dist0 = spatial.distance.cosine(v_sentence, midpoint0)
        dist1 = spatial.distance.cosine(v_sentence, midpoint1)
        
        #a low ratio makes it 'easier' to return False...
        #a high ratio does the inverse
        if dist0 < dist1:
            return False
        else:
            return True


    def get_affordance(self, word, num_matches=5):
        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            #v_word = self.scholar.model[word]
            v_word = self.word_encode(word)
        elif isinstance(word, (np.ndarray, np.generic) ):
            v_word = word
        else:
            print("ERROR: unknown data type for Strategist.get_affordance():")
            print(type(effect))
            sys.exit()

        if isinstance(word,basestring):
             if word[-3:] == '_NN':
                v_result = v_word - self.canonical_vectors['affordance']
             elif word[-3:] == '_VB':
                v_result = v_word + self.canonical_vectors['affordance']
             else:
                raise ValueError("Hey, I thought affordance parameters were supposed to be either nouns or verbs...")
        else:
            v_result = v_word + self.canonical_vectors['affordance']
        return self.closest_words(v_result, num_matches, return_distance=True)


    def get_cause(self, effect, stddev=None):
        #if the effect is a string, convert it to a vector
        if isinstance(effect, basestring):
            v_effect = self.encode(effect)
        elif isinstance(effect, (np.ndarray, np.generic) ):
            v_effect = effect
        else:
            print("ERROR: unknown data type for Strategist.get_cause():")
            print(type(effect))
            sys.exit()

        v_cause = v_effect - self.canonical_vectors['causation']
        print(self.jitter(v_cause,stddev=stddev))
        return self.closest_sentences(v_cause)
        

    def get_effect(self, cause, stddev=None):
        #if the cause is a string, convert it to a vector
        #if type(cause) == type('str'):
        if isinstance(cause, basestring):
            v_cause = self.encode(cause)
        ##elif type(cause) == type(np.array([0,5,7,7])):
        elif isinstance(cause, (np.ndarray, np.generic) ):
            v_cause = cause
        else:
            print("ERROR: unknown data type for Strategist.get_cause():")
            print(type(cause))
            sys.exit()

        #v_cause = self.encode(cause)
        v_effect = v_cause + self.canonical_vectors['causation']
        print(self.jitter(v_effect, stddev=stddev))
        return self.penseur.closest_sentences(v_effect)


    def test(self):
        for key in self.canonical_examples:
            print(key)
            print(self.canonical_examples[key])

    def tag_analogies(self,analogies):
        new_analogies=[]
        for analogy in analogies[1:]:
            new_analogy=[]
            for word in analogy:
                tag = self.scholar.get_most_common_tag(word)
                new_analogy.append(word + "_" + tag)
            new_analogies.append(new_analogy)

        return new_analogies

    def CoRL_run_pisa_tests(self,output_file,limit=100000,unpisa=True,superpisa=False):

        NUM_MATCHES=30        

        corpus_list = ['accessing_containers.p','affordance.p','belong.p','causation.p','containers.p','rooms_for_containers.p','locations_for_objects.p','rooms_for_objects.p','tools.p','trash_or_treasure.p','travel.p','causation_VBZ.p']
        abbrev_list=['accessing containers','affordance','belong','causation','containers','rooms for containers', 'locations for objects','rooms for objects','tools','trash or treasure','travel','causation_VBZ']
       # corpus_list=['tagged_analogy_subcorp7_opposites.p','tagged_analogy_subcorp3_countries_currency.p','tagged_analogy_subcorp12_past_tense.p','tagged_analogy_subcorp5_family_relations.p','tagged_analogy_subcorp1_capitals_countries.p','tagged_analogy_subcorp2_capitals_world.p','tagged_analogy_subcorp4_city_state.p','tagged_analogy_subcorp6_adj_adverb.p','tagged_analogy_subcorp8_comparative.p','tagged_analogy_subcorp9_superlative.p','tagged_analogy_subcorp10_present_participle.p','tagged_analogy_subcorp11_nationality_adj.p','tagged_analogy_subcorp13_plural.p','tagged_analogy_subcorp14_plural_verbs.p']
        #abbrev_list=['subcorp7','subcorp3','subcorp12','subcorp5','subcorp1','subcorp2','subcorp4','subcorp6','subcorp8','subcorp9','subcorp10','subcorp11','subcorp13','subcorp14']

        #scale_factors = [0.1,0.5,1.0,2.0,5.0]
        #scale_factors = [1.0,0.5]
        #scale_factors = [0.5,1.0,1.5]
        scale_factors = [0.3]
        
        return_dict = {}
        return_dict['normal_exclude']={}
        return_dict['normal']={}
        return_dict['word2vec_analogy']={}
        for s in scale_factors:
            return_dict['pisa ' + str(s)] = {}
        
        vectors = self.get_scholar_vectors()
        file_counter = 0
        for corpus,abbrev in zip(corpus_list,abbrev_list):
            file_counter+=1
            print( corpus )
            print( abbrev )
            with open('corrected_corpora/' + corpus) as f:
		if (sys.version_info[0] >= (3.0)):
                	analogies_unfiltered=pickle.load(f, encoding='latin1')
		else:
                	analogies_unfiltered=pickle.load(f)
                analogies=[]
                for a in analogies_unfiltered:
                    if self.scholar.exists_in_model(a[0]) and self.scholar.exists_in_model(a[1]) and self.scholar.exists_in_model(a[2]) and self.scholar.exists_in_model(a[3]):
                        analogies.append(a)
                    else:
                        print( "DROPPING ANALOGY: word not in model." )
                        print( a )

            NUM_CANONICAL_EXAMPLES = 10
                
            #Take the first n analogies as our direction of travel
            print( "\nCOMPILING CANONICAL EXAMPLES" )
            canonical_examples=[]
            for a in analogies[:NUM_CANONICAL_EXAMPLES]:
                example = [a[0],a[1]]

                if example not in canonical_examples:
                    print( example )
                    canonical_examples.append(example)
                example = [a[2],a[3]]
                if example not in canonical_examples:
                    print( example )
                    canonical_examples.append(example)

            canonical_vector = np.zeros(len(self.word_encode(analogies[0][0])))
            target_point = np.zeros(len(self.word_encode(analogies[0][0])))
            for ex in canonical_examples:
                canonical_vector += self.word_encode(ex[1]) - self.word_encode(ex[0])
                target_point += self.word_encode(ex[1])

            canonical_vector = canonical_vector / len(canonical_examples)
            target_point = target_point / len(canonical_examples)
            print( "Dividing by " + str(len(canonical_examples)) )
            print( "Canonical vector length is " + str(np.linalg.norm(canonical_vector)) )

            #Quick hack for recordkeeping purposes: fold canonical vector length into canonical examples (does not affect data run since canonical vector has already been calculated.)
            canonical_examples.append(np.linalg.norm(canonical_vector))

            for s in scale_factors:
                self.alpha=s
                print( str(s) )
                scores=[0,0,0,0,0]
                distance_to_correct_answer=[]
                found_words_list=[]
                results = []

                print( corpus )
                print( len(analogies) )
                if len(analogies)<1:
                    print( "ERROR READING INPUT FILE: " + 'corrected_corpora/'+corpus )
                for a in tqdm(analogies[:limit]):
                    #EXTRACT THE DIRECTION OF TRAVEL
                    #direction_of_travel=self.word_encode(a[1])-self.word_encode(a[0])
                    direction_of_travel = canonical_vector

                    #distances=self.get_pisa_scores(vectors,self.word_encode(a[2]),direction_of_travel)           
                    if unpisa==True:
                        distances=self.get_unpisa_scores(vectors,self.word_encode(a[2])+direction_of_travel,direction_of_travel)           
                    elif superpisa==True:
                        distances=self.get_superpisa_scores(vectors,self.word_encode(a[2])+direction_of_travel,direction_of_travel,target_point)           
                    else:
                        distances=self.get_pisa_scores(vectors,self.word_encode(a[2])+direction_of_travel,direction_of_travel)           

                    found_words = []
                    found_distance = []
                    for count in range(0, NUM_MATCHES):
                            #min_dist = min(distances)
                            #index = distances.tolist().index(min_dist)
                            #min_dist = np.amin(distances)
                            min_dist = np.nanmin(distances)
                            index = np.where(distances==min_dist)[0][0]
                            found_words.append(self.scholar.model.vocab[index])
                            found_distance.append(min_dist)
                            distances[index] = 1000

                    if a[3] == found_words[0]:
                        scores[0] += 1
                    elif a[3] in found_words[:5]: 
                           scores[1] += 1
                    elif a[3] in found_words[:15]: 
                           scores[2] += 1
                    elif a[3] in found_words[:30]: 
                           scores[3] += 1
                    else:
                        scores[4] += 1

                    b=a[:]
                    b[3]=found_words[0]
                    results.append(b)
                    distance_to_correct_answer.append(spatial.distance.cosine(self.word_encode(a[3]),self.word_encode(b[3])))
                    found_words_list.append(found_words[:30])

                return_dict['pisa ' + str(s)][abbrev] = [[],[],[],[],[]]
                return_dict['pisa ' + str(s)][abbrev][0] = scores
                return_dict['pisa ' + str(s)][abbrev][1] = distance_to_correct_answer
                return_dict['pisa ' + str(s)][abbrev][2] = results
                return_dict['pisa ' + str(s)][abbrev][3] = found_words_list
                return_dict['pisa ' + str(s)][abbrev][4] = canonical_examples

            #Now generate the true mikolov behavior,
            #for comparison...
            scores=[0,0,0,0,0]
            distance_to_correct_answer=[]
            found_words_list=[]
            results = []
            for a in tqdm(analogies[:limit]):
                self.scholar.number_analogy_results=NUM_MATCHES
                found_words = self.scholar.analogy(a[1] + ' -' + a[0] + ' ' + a[2])

                if a[3] == found_words[0]:
                    scores[0] += 1
                elif a[3] in found_words[:5]: 
                    scores[1] += 1
                elif a[3] in found_words[:15]: 
                    scores[2] += 1
                elif a[3] in found_words[:30]: 
                    scores[3] += 1
                else:
                    scores[4] += 1
                
                b=a[:]
                b[3]=found_words[0]
                results.append(b)
                distance_to_correct_answer.append(spatial.distance.cosine(self.word_encode(a[3]),self.word_encode(b[3])))
                found_words_list.append(found_words[:30])

            return_dict['word2vec_analogy'][abbrev] = [[],[],[],[],[]]
            return_dict['word2vec_analogy'][abbrev][0] = scores
            return_dict['word2vec_analogy'][abbrev][1] = distance_to_correct_answer
            return_dict['word2vec_analogy'][abbrev][2] = results
            return_dict['word2vec_analogy'][abbrev][3] = found_words_list
            return_dict['word2vec_analogy'][abbrev][4] = canonical_examples
            

            #Now calculate normal exclude on the canonical vector
            scores=[0,0,0,0,0]
            results = []
            distance_to_correct_answer=[]
            found_words_list=[]
            for a in tqdm(analogies[:limit]):
                found_words = self.closest_words_fast(self.word_encode(a[2])+canonical_vector,num_matches=NUM_MATCHES+3)

                for word in a[:3]:
                   if word in found_words:
                        found_words.remove(word)
                found_words=found_words[:NUM_MATCHES]

                if a[3] == found_words[0]:
                    scores[0] += 1
                elif a[3] in found_words[:5]: 
                    scores[1] += 1
                elif a[3] in found_words[:15]: 
                    scores[2] += 1
                elif a[3] in found_words[:30]: 
                    scores[3] += 1
                else:
                    scores[4] += 1
                
                b=a[:]
                b[3]=found_words[0]
                results.append(b)
                distance_to_correct_answer.append(spatial.distance.cosine(self.word_encode(a[3]),self.word_encode(b[3])))
                found_words_list.append(found_words[:30])

            return_dict['normal_exclude'][abbrev] = [[],[],[],[],[]]
            return_dict['normal_exclude'][abbrev][0] = scores
            return_dict['normal_exclude'][abbrev][1] = distance_to_correct_answer
            return_dict['normal_exclude'][abbrev][2] = results
            return_dict['normal_exclude'][abbrev][3] = found_words_list
            return_dict['normal_exclude'][abbrev][4] = canonical_examples
            
            #And finally, calculate normal exclude on the canonical vector
            scores=[0,0,0,0,0]
            results=[]
            distance_to_correct_answer=[]
            found_words_list=[]
            for a in tqdm(analogies[:limit]):
                found_words = self.closest_words_fast(self.word_encode(a[2])+canonical_vector,num_matches=NUM_MATCHES)

                #NOT NEEDED for normal-no-exlcude
                #for word in a[:3]:
                #   if word in found_words:
                #        found_words.remove(word)
                #found_words=found_words[:100]

                if a[3] == found_words[0]:
                    scores[0] += 1
                elif a[3] in found_words[:5]: 
                    scores[1] += 1
                elif a[3] in found_words[:15]: 
                    scores[2] += 1
                elif a[3] in found_words[:30]: 
                    scores[3] += 1
                else:
                    scores[4] += 1
                
                b=a[:]
                b[3]=found_words[0]
                results.append(b)
                distance_to_correct_answer.append(spatial.distance.cosine(self.word_encode(a[3]),self.word_encode(b[3])))
                found_words_list.append(found_words[:30])

            return_dict['normal'][abbrev] = [[],[],[],[],[]]
            return_dict['normal'][abbrev][0] = scores
            return_dict['normal'][abbrev][1] = distance_to_correct_answer
            return_dict['normal'][abbrev][2] = results
            return_dict['normal'][abbrev][3] = found_words_list
            return_dict['normal'][abbrev][4] = canonical_examples

            f=open('dict_files/temp' + str(file_counter) + '.p','wb')
            pickle.dump(return_dict,f)
            f=open('dict_files/temp' + str(file_counter) + '.p','wb')
            pickle.dump(return_dict,f)

        f=open(output_file, 'wb')
        pickle.dump(return_dict,f)
        return return_dict

    def CoRL_analogy(self,word,tag,num_matches=15):
        
        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            v_word = self.word_encode(word)
        else:
            v_word = word
        
        v_result = v_word + self.canonical_vectors[tag]
        
        normal_words,normal_scores = self.closest_words_fast(v_result, num_matches=num_matches+1, return_distance=True)
        if word in normal_words:
            normal_words.remove(word)

        pisa_words,pisa_scores = self.closest_pisa_words(v_result, num_matches=num_matches, analogy_vector=self.canonical_vectors[tag], return_distance=True)

        unpisa_words,unpisa_scores = self.closest_unpisa_words(v_result, num_matches=num_matches, analogy_vector=self.canonical_vectors[tag], return_distance=True)

        superpisa_words,superpisa_scores = self.closest_superpisa_words(v_result, num_matches=num_matches, analogy_vector=self.canonical_vectors[tag],target_point=self.canonical_target_points[tag], return_distance=True)

        #create a probability estimate for each word in the result set
        normal_probs = ((2. - np.array(normal_scores))/np.sum(2. - np.array(normal_scores))).tolist()
        pisa_probs =   ((4. - np.array(pisa_scores))/np.sum(4. - np.array(pisa_scores))).tolist()
        
        print( "\nNORMAL RESULTS:" )
        #string=''
        #for word,prob in zip(normal_words,normal_probs):
        #    string += word + " " + str(round(prob*100, 1)) + '%, '
        #print string
        print(', '.join(normal_words))

        print( "\nPISA RESULTS:" )
        #string = ''
        #for word,prob in zip(pisa_words,pisa_probs):
        #    string += word + " " + str(round(prob*100, 1)) + "%, "
        #print string
        print(', '.join(pisa_words))

        print( "\nUNPISA RESULTS:" )
        print( ', '.join(unpisa_words) )

        print( "\nSUPERPISA RESULTS:" )
        print( ', '.join(superpisa_words) )

    def CoRL_categorize(self,word,tag,categories):
 
        #if the word is a string, convert it to a vector
        if isinstance(word, basestring):
            v_word = self.word_encode(word)
        else:
            v_word = word

        v_result = v_word + self.canonical_vectors[tag]

        distances=[]
        for c in categories:
            distances.append(spatial.distance.cosine(v_word,self.word_encode(c)))

        print( categories )
        print( distances )

        min_index = np.argmin(distances)
        return categories[min_index]

