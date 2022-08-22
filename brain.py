import strategist
import scholar
import numpy as np
import random as rand
import penseur
import re
import nltk
from scipy import spatial
import actionMemory
import sys

class Brain():

    def __init__(self):
        self.strategist = strategist.Strategist(slim=False,intentional_agent=True)
        self.scholar = self.strategist.scholar

        #self.verbs_for_noun = [] #verbs are removed from the list once
                                 #tried, thus avoiding repetition until
                                 #all verbs have been attempted once
        #self.prepositionl_phrases_for_verb = [] #same as verbs_for_noun
        self.verbose=False

        #MENTAL STATE
        self.frustration={}
        self.max_frustration_level = 2
        ##!self.max_frustration_level = 2
        self.navigation_goal = -1
        
        self.actionMemory = actionMemory.ActionMemory()
        self.valid_transitions = {}

        self.related_noun_threshold = .45  #arbitrarily chosen based on observations
        #SET FRUSTRATION VALUES
        self.frustration_values={}
        self.frustration_values[0] = {}
        #self.frustration_values[0]['noun_cutoff'] = 2
        #self.frustration_values[0]['num_verbs'] = 30
        #self.frustration_values[0]['chance_of_two_word_combination'] = 0
        #self.frustration_values[0]['filter_verbs'] = True
        #self.frustration_values[0]['frustration_list'] = False
        #self.frustration_values[0]['noun_cutoff'] = 0
        self.frustration_values[0]['noun_cutoff'] = 0
        self.frustration_values[0]['num_verbs'] = 10
        self.frustration_values[0]['chance_of_two_word_combination'] = 0
        self.frustration_values[0]['filter_verbs'] = True
        self.frustration_values[0]['frustration_list'] = False
        self.frustration_values[0]['frustration_nav_list'] = False
        
        self.frustration_values[1] = {}
        self.frustration_values[1]['noun_cutoff'] = 0.1
        self.frustration_values[1]['num_verbs'] = 30
        self.frustration_values[1]['chance_of_two_word_combination'] = 20
        self.frustration_values[1]['filter_verbs'] = True
        self.frustration_values[1]['frustration_list'] = True
        self.frustration_values[1]['frustration_nav_list'] = False

        self.frustration_values[2] = {}
        self.frustration_values[2]['noun_cutoff'] = 2
        self.frustration_values[2]['num_verbs'] = 30
        self.frustration_values[2]['chance_of_two_word_combination'] = 50
        self.frustration_values[2]['filter_verbs'] = False
        self.frustration_values[2]['frustration_list'] = True
        self.frustration_values[2]['frustration_nav_list'] = True

        #self.noun_cutoff=2
        self.prereq_cutoff=2 

        #CACHING
        self.cache_extract_targeted_text = {}    
        self.cache_choose_object = {}
        self.cache_possible_actions = {}
        self.cache_find_objects = {}
        self.cache_verbs_for_noun = {}
        #self.cache_state_object_verbs_for_noun = {}
        self.verb_try_list = {} #caches state/obj/verbs_for_noun

        #NAVIGATION ESSENTIALS
        self.navigation_list =['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'up', 'down', 'enter', 'exit', 'look', 'inventory']
        self.frustrated_navigation =['yes','no','hint']
        #drop shows up as an option only if the agent is frustrated
        #self.essential_manipulation_list = ['examine', 'push', 'pull', 'enter']
        #self.frustration_list = ['drop', 'get', 'open', 'close']
        self.essential_manipulation_list = ['open','get','examine','turn on']
        self.frustration_list = ['drop', 'close', 'turn off','open','push','pull','turn on', 'turn off', 'enter', 'wear', 'remove', 'enter']

        self.manipulation_list = ['throw', 'spray', 'stab', 'slay', 'open', 'pierce', 'thrust', 'exorcise', 'place', 'jump', 'take', 'make', 'read', 'strangle', 'swallow', 'slide', 'wave', 'look', 'dig', 'pull', 'put', 'rub', 'fight', 'ask', 'score', 'apply', 'take', 'knock', 'block', 'kick', 'step', 'break', 'wind', 'blow', 'crack', 'drop', 'blast', 'leave', 'yell', 'skip', 'stare', 'hurl', 'hit', 'kill', 'glass', 'engrave', 'bottle', 'pour', 'feed', 'hatch', 'swim', 'spray', 'melt', 'cross', 'insert', 'lean', 'sit', 'move', 'fasten', 'play', 'drink', 'climb', 'walk', 'consume', 'kiss', 'startle', 'shout', 'close', 'cast', 'set', 'drive', 'lift', 'strike', 'startle', 'catch', 'board', 'speak', 'think', 'get', 'answer', 'tell', 'feel', 'get', 'turn', 'listen', 'read', 'watch', 'wash', 'purchase', 'do', 'sleep', 'fasten', 'drag', 'swing', 'empty', 'switch', 'slip', 'twist', 'shoot', 'slice', 'read', 'burn', 'hop', 'rub', 'ring', 'swipe', 'display', 'scrub', 'hug', 'operate', 'touch', 'sit', 'sweep', 'fix', 'walk', 'crack', 'skip']
        self.manipulation_list += ['wait', 'point', 'light', 'unlight', 'use', 'ignite', 'wear', 'remove', 'unlock', 'lock', 'examine', 'inventory', '']
                

    def refresh(self):
        self.frustration={}
        self.cache_extract_targeted_text = {}    
        self.cache_choose_object = {}
        self.cache_possible_actions = {}
        self.cache_find_objects = {}
        self.cache_verbs_for_noun = {}
        self.verb_try_list = {} #caches state/obj/verbs_for_noun
        self.actionMemory = actionMemory.ActionMemory()
        self.valid_transitions = {}


    def update(self, current_state=None, inventory_items=None):
        if current_state is not None:
            self.current_state = current_state
        if inventory_items is not None:
            self.inventory_items=inventory_items        
            for state in self.verb_try_list.keys():
                frustration = self.frustration[state]
                num_matches = self.frustration_values[frustration]['num_verbs']
                for o in inventory_items:
                    if o not in self.verb_try_list[state].keys():
                        verbs = self.verbs_for_noun(o,num_matches=num_matches)
                        self.verb_try_list[state][o] = verbs

    def categorize(self, processed_text, text_vectors, tag):
        sentences = []
        for i in range(len(processed_text)):
            line = processed_text[i]
            v_line = text_vectors[i]
            if self.strategist.categorize(v_line, tag):
                sentences.append(line)
        return sentences

    def extract_targeted_text(self, game_text, tags):

        if game_text not in self.cache_extract_targeted_text.keys():
            self.cache_extract_targeted_text[game_text] = {}
            
        return_vals = []
        processed_text = []
        text_vectors=[]
        for line in re.split(r'[.!?]+ *',game_text):
            line = line.strip()
            if line!='':
                processed_text.append(line)
                text_vectors.append(self.strategist.encode(line))

        for t in tags:
            if t in self.cache_extract_targeted_text[game_text].keys():
                return_vals.append(self.cache_extract_targeted_text[game_text][t])
            else:
                return_vals.append(self.categorize(processed_text, text_vectors, t))
                self.cache_extract_targeted_text[game_text][t] = return_vals[-1]    

        return return_vals


    def prioritize_phrases(self, phrases):
        #prioritizes the most 'likely' phrases first

        #print "Prioritizing prepositional phrases..."
        priorities = []
        for p in phrases:
            v_p = self.strategist.encode(p)
            priorities.append(self.strategist.canonical_projection(v_p,'ia_good_prepositions'))

        return [phrase for (phrase,priority) in sorted(zip(phrases,priorities))]
   
    def prioritize_phrases_alt(self, phrases):
        #prioritizes the most 'likely' phrases first

        if self.verbose==True:
           print("Prioritizing prepositional phrases...")
        phrase_vectors=[]
        for p in phrases:
            v_p = self.strategist.encode(p)
            phrase_vectors.append(v_p)
        priorities = self.strategist.canonical_multiprojection(np.array(phrase_vectors),'ia_good_prepositions')
        return [phrase for (phrase,priority) in sorted(zip(phrases,priorities))]
 
    def prep_phrases(self, X, Y):
        phrases = []
        phrases.append('put ' + X + ' on ' + Y)
        phrases.append('put ' + X + ' in ' + Y)
        phrases.append('put ' + X + ' inside ' + Y)
        phrases.append('put ' + X + ' beneath ' + Y)
        phrases.append('give ' + X + ' to ' + Y)
        phrases.append('take ' + X + ' from ' + Y)
        phrases.append('remove ' + X + ' from ' + Y)
        phrases.append('use ' + X + ' on ' + Y)
        
        v_aff = self.strategist.get_relation(self.strategist.tag(X.lower().split(' ')[-1]),'affordance')
        verbs = self.strategist.strip_tags(self.strategist.closest_words_fast(v_aff,num_matches=5))
        for v in verbs:
            if v.lower()[-1] != X.lower()[-1]:
                phrases.append(v + ' with ' + X)
                phrases.append(v + ' with ' + Y)
                phrases.append(v + ' on ' + X)
                phrases.append(v + ' on ' + Y)
                phrases.append(v + ' inside ' + X)
                phrases.append(v + ' inside ' + Y)
                phrases.append(v + ' beneath ' + X)
                phrases.append(v + ' beneath ' + Y)
                phrases.append(v + ' ' + X + ' with ' + Y)
                phrases.append(v + ' ' + Y + ' with ' + X)

        return phrases

    def prepositional_combinations_with_noun(self,obj,noun):
        phrases = []
        phrases = phrases + self.prep_phrases(obj,noun)
        if (obj != noun):
            phrases = phrases + self.prep_phrases(noun,obj)
        return phrases

    def prepositional_combinations_with_verb(self,untagged_obj,tagged_verb,available_objects):
        phrases = []

        nouns = self.find_verb_noun_matches(tagged_verb,available_objects)
        verb = self.strategist.untag(tagged_verb)
        for noun in nouns:
            if noun.lower()[-1] != verb.lower()[-1]:
                phrases.append(verb + ' ' + untagged_obj + ' with ' + noun)
                phrases.append(verb + ' with ' + noun)

        return phrases

    def combine_objects(self,obj,hint,memory,current_state,inventory_items):
        verbs = self.verbs_for_noun(obj)
        if len(verbs) > 0:
            tagged_verb = rand.choice(verbs) + '_VB'
            available_objects = self.find_objects(memory[current_state])+inventory_items
            prepositional_phrases = self.prepositional_combinations_with_verb(obj,tagged_verb,available_objects)
        else:
            return []
        return prepositional_phrases

    def combine_objects_using_verb(self,obj,hint,memory,current_state,inventory_items):
        #try to find and interact with related objects
        available_nouns = self.find_objects(memory[current_state])+inventory_items
        compatible_nouns = self.find_noun_matches(obj,available_nouns)
        if obj in compatible_nouns:
            compatible_nouns.remove(obj)
        if len(compatible_nouns) > 0:
            noun = rand.choice(compatible_nouns)
            if self.verbose==True:
                print("Calculating prepositional phrases for " + obj + ' and ' + noun)
            prepositional_phrases = self.prepositional_combinations(obj,noun)
            if self.verbose==True:
                print(prepositional_phrases)
            return prepositional_phrases
        else:
            return []

 
    def filter_verbs_for_noun(self, obj, verbs):
        filtered_verbs = []
        distances = []

        for v in verbs:
            if v in self.manipulation_list + self.essential_manipulation_list:
                filtered_verbs.append(v)        

        #for v in verbs:
        #    tagged_verb = self.strategist.tag(v)
        #    tagged_object = self.strategist.tag(obj)
        #    vector = self.strategist.word_encode(tagged_verb) - self.strategist.word_encode(tagged_object)
        #    distance = spatial.distance.cosine(vector, self.strategist.canonical_vectors['affordance'])
        #    distances.append(distance)
        #    if distance < threshold:
        #        filtered_verbs.append(v)

        #print "DISTANCES:"
        #print distances
        return list(set(filtered_verbs))

    def verbs_for_noun(self,obj,num_matches=15,logic_filter=False, frustration_list = False):

        #THIS CACHING WAS BADLY IMPLEMENTED
        #TAKE IT OUT FOR NOW
        #if obj not in self.cache_verbs_for_noun.keys():
        #    self.cache_verbs_for_noun[obj] = {}
        #if num_matches in self.cache_verbs_for_noun[obj].keys():
        #    return self.cache_verbs_for_noun[obj][num_matches]

        o = obj
        if len(o.split(' ')) > 0:
            base_object = o.split(' ')[-1]
        else:
            base_object = o
        if self.scholar.exists_in_model(base_object.lower()+"_NN"):
            v_aff = self.strategist.get_relation(base_object.lower()+"_NN",'affordance',num_matches=num_matches)
            verbs = self.strategist.strip_tags(self.strategist.closest_words_fast(v_aff,num_matches=num_matches))
        else:
            verbs = []

        verbs = list(set(verbs)) 
        if base_object in verbs:
            verbs.remove(base_object)

        verbs += self.essential_manipulation_list
        if frustration_list == True:
            verbs += self.frustration_list

        new_verbs = []
        for v in verbs:
            if not self.actionMemory.is_invalid(v + ' ' + obj):
                new_verbs.append(v)
        verbs = new_verbs

        if logic_filter == True:
            print("filtering verbs...")
        
            #Well, THIS method didn't work worth beans.
            #Let's try something else.
            #filtered_verbs = []
            #for v in verbs:
            #    if self.strategist.categorize(v + ' ' + obj, 'ia_logical_verbnoun'):
            #            filtered_verbs.append(v)
            #verbs = filtered_verbs

            verbs = self.filter_verbs_for_noun(obj,verbs)

        else:
            rand.shuffle(verbs)

        #self.cache_verbs_for_noun[obj][num_matches] = verbs
        return verbs
        

    def match_to_list(self, desired_nouns, available_nouns):

        if len(desired_nouns) == 0 or len(available_nouns) == 0:
            return []

        related_vectors=[]
        for n in desired_nouns:
            related_vectors.append(self.strategist.word_encode(n))

        min_distances=[]
        closest_word=[]

        #For each available object, find its minimum distance
        #to any of the desired objects...
        for w in available_nouns:
            base_word = w.split(' ')[-1].lower()
            #print base_word
            base_word = self.strategist.tag(base_word)
            #base_word = base_word + '_NN'
            if not self.scholar.exists_in_model(base_word):
                min_distances.append(2)
                closest_word.append('')
            else:
                v_base = self.strategist.word_encode(base_word)
                distances = 1-np.clip(np.dot(related_vectors,v_base.T),0,1) #no normalization needed b.c. they're all unit vectors
                min_distances.append(np.amin(distances))
                closest_word.append(desired_nouns[np.argmin(distances)])

        #THRESHOLD = 0.39   #arbitrarily chosen based on observations
        THRESHOLD = self.related_noun_threshold   #arbitrarily chosen based on observations
        matching_nouns=[]
        for w in range(len(available_nouns)):
            if min_distances[w] < THRESHOLD:
                matching_nouns.append(available_nouns[w])

        return self.strategist.strip_tags(matching_nouns)

    def find_verb_noun_matches(self,tagged_verb,available_nouns):
        if not self.scholar.exists_in_model(tagged_verb):
            return []
        v_aff = self.strategist.get_relation(tagged_verb,'affordance',invert=True)
        related_nouns = self.strategist.closest_words_fast(v_aff)
        if self.verbose==True:
            print("DESIRED NOUNS: ")
            print(related_nouns)
        return self.match_to_list(related_nouns, available_nouns)

    def find_noun_matches(self,obj,available_nouns):
        noun = obj.lower().split(' ')[-1]
        tagged_noun = self.strategist.tag(noun)
        tagged_noun = tagged_noun+'NN'
        if self.scholar.exists_in_model(tagged_noun):
            v_belong = self.strategist.get_relation(tagged_noun,'belong')
            related_nouns = self.strategist.closest_words_fast(v_belong)
            return self.match_to_list(related_nouns, available_nouns)
        else:
            return []


    ################## TEXT PARSING #####################

    def find_objects(self, game_text, two_word_objects=True, prioritize=False, cutoff=2): 

        if game_text == '' or game_text is None:
            return []

        #caching
        if cutoff not in self.cache_find_objects.keys():
            self.cache_find_objects[cutoff] = {}
        if game_text in self.cache_find_objects[cutoff].keys():
            return self.cache_find_objects[cutoff][game_text]

        tokens = nltk.word_tokenize(game_text)
        tags = nltk.pos_tag(tokens)
        nouns = [word for word,pos in tags if word.isalnum() and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

        #If we are using two-word nouns, then select them
        #first and prioritize afterward
        if two_word_objects == True:
            tokens = nltk.word_tokenize(game_text)
            tags = nltk.pos_tag(tokens)
            for i in range(0, len(tags) - 1):
                if (tags[i][1] == "JJ") and (tags[i+1][1] in ["NN", "NNP", "NNS", "NNPS"]):
                    nouns.append(tags[i][0] + " " + tags[i+1][0])

        #remove duplicates
        nouns = list(set(nouns))

        #Prioritize nouns 
        if prioritize == True and len(nouns) > 1:
            #get the w2v vectors for each object
            data = []
            for nn in nouns:
                n = nn.split(' ')[-1] #account for two-word nouns
                try:
                    tagged_noun = n.lower() + '_' + self.scholar.get_most_common_tag(n.lower())
                    #tagged_noun = n.lower() + '_NN'
                except:
                    tagged_noun = n
                    
                if self.scholar.exists_in_model(tagged_noun):
                    #print "appending " + nn + ' as ' + tagged_noun
                    data.append(self.scholar.model[tagged_noun])
                else:
                    #print "ignoring " + nn + " as " + tagged_noun
                    data.append(np.mean(self.scholar.model.vectors, axis=0))

            vector=self.strategist.canonical_vectors['manipulability']
            flattened_data = np.dot(vector, np.array(data).T).T
            #print flattened_data

            zipped = zip(flattened_data, nouns)
            if sys.version_info[0] >= (3.0):
                sorted(zipped)
            else:
                zipped.sort()
            sorted_nouns = [nouns for (flattened_data, nouns) in zipped]
            sorted_distances = [flattened_data for (flattened_data, nouns) in zipped]
            #print("FIND OBJECTS:")
            #print( sorted_nouns)
            #print(sorted_distances)

            nouns=sorted_nouns

            if False:
            #if len(nouns) > 3:
                cutoff_nouns = []
                for i in range(len(nouns)):
                    if sorted_distances[i] <= cutoff:
                        cutoff_nouns.append(sorted_nouns[i])

                nouns = cutoff_nouns

        self.cache_find_objects[cutoff][game_text] = nouns
        return nouns

    def populate_verb_try_list(self, current_state, available_objects, inventory_items):
        if self.verbose == True:
            print("POPULATING try_list for state " + str(current_state))

        if current_state not in self.verb_try_list.keys():
            self.verb_try_list[current_state] = {}

        #We can safely assume that self.frustration[current_state]
        #has already been initialized
        frustration = self.frustration[current_state]
        num_matches = self.frustration_values[frustration]['num_verbs']
        filter_verbs = self.frustration_values[frustration]['filter_verbs']
        frustration_list = self.frustration_values[frustration]['frustration_list']

        for o in available_objects+inventory_items:
            verbs = self.verbs_for_noun(o,num_matches = num_matches, logic_filter=filter_verbs, frustration_list = frustration_list)

            self.verb_try_list[current_state][o] = verbs
        
            if self.verbose == True:
                print( o )
                print( self.verb_try_list[current_state][o] )


    def possible_actions(self,game_text, required_verbs, inventory_items=[]):
        if game_text == '' or game_text is None:
            return []

        if game_text in self.cache_possible_actions.keys():
            return self.cache_possible_actions[game_text]

        objects = self.find_objects(game_text, two_word_objects=True)

        try_list = []
        for o in objects:
            if len(o.split(' ')) > 0:
                base_object = o.split(' ')[-1]
            else:
                base_object = o
            if self.scholar.exists_in_model(base_object.lower()+"_NN"):
                v_aff = self.strategist.get_relation(base_object.lower()+"_NN",'affordance',num_matches=15)
                words = self.strategist.closest_words_fast(v_aff)
            else:
                words = []
            ##!pisa_words = self.strategist.closest_pisa_words(v_aff,metric='affordance')
            #print "\nWORDS FOR '" + o + "'"
            #print words
            ##!print pisa_words

            for w in self.strategist.strip_tags(words) + required_verbs:
                if w != base_object:
                    try_list.append(w + ' ' + o)

        final_list = list(set(try_list))
        self.cache_possible_actions[game_text] = final_list
        return final_list


    def prerequisites(self, goal, actions, action_vectors, use_pisa=False,cutoff=-1):
        #uses skipthought vectors to prioritize possible actions
        #according to goal prerequisites
	if cutoff == -1:
            cutoff = self.prereq_cutoff #default is 2
	    if use_pisa==True:
		cutoff=cutoff*2

        analogy_vector = self.strategist.canonical_vectors['subgoal_for_goal']

        v_prereq = self.strategist.encode(goal) + analogy_vector
        if self.verbose == True:
            prereqs, distances = self.strategist.proximity_sort(v_prereq,action_vectors,actions, cutoff=cutoff, return_distance=True, use_pisa=use_pisa, pisa_vector=analogy_vector)
            if use_pisa == True:
                print("Pisa vector:")
                print(analogy_vector)
            return prereqs
        else:
            prereqs = self.strategist.proximity_sort(v_prereq,action_vectors,actions, cutoff=cutoff, use_pisa=use_pisa, pisa_vector=analogy_vector)
            return prereqs

    def interact_with_object(self,obj):
        pass

    def objectless_action(self, memories, current_state, inventory_items):
        return []

    def navigation_actions(self, potential_locations):
        return []

    def choose_a_goal(self):
        return raw_input("What is my goal?")

    def choose_behavior(self, memories, current_state, inventory_items):

        if current_state not in self.frustration.keys():
            self.frustration[current_state] = 0 #frustration always defaults to 0
        frustration = self.frustration[current_state]
        if frustration == 0:
            #WARNING: THE LINE BELOW WILL FAIL IF memories IS EMPTY
            #self.navigation_goal = rand.choice(memories.keys())
            #return rand.choice(['PLAY','PLAY','PLAY','PLAY','EXPLORE','NAVIGATE'])
            return 'PLAY'
        if frustration == 1:
            return rand.choice(['PLAY','PLAY','PLAY','PLAY','PLAY','EXPLORE'])
        if frustration == 2:
            return rand.choice(['PLAY','PLAY','EXPLORE'])

    def choose_object(self, memories, current_state, inventory_items):
        #print("CHOOSE OBJECT")
        #print(current_state)
        if current_state == -1:
            print("\nABORTING OBJECT SELECTION: no game text found")
            return ''

        if current_state not in memories.keys():
            print("\nABORTING OBJECT SELECTION: no game text found")
            return ''

        game_text = memories[current_state]
        #print(memories[current_state])

        #cutoff = self.noun_cutoff
        if current_state not in self.frustration.keys():
            self.frustration[current_state] = 0 #frustration level always begins at 0
        frustration = self.frustration[current_state]
        print("frustration level for state " + str(current_state) + " is " + str(frustration))
        cutoff = self.frustration_values[frustration]['noun_cutoff']
        #print("cutoff for " + str(current_state) + " is " + str(cutoff))

        if game_text in self.cache_choose_object.keys():
            if cutoff in self.cache_choose_object[game_text]:
                available_objects = self.cache_choose_object[game_text][cutoff]
            else:
                available_objects = self.find_objects(game_text,cutoff=cutoff)
                self.cache_choose_object[game_text][cutoff] = available_objects
                self.populate_verb_try_list(current_state, available_objects, inventory_items)
        else:
            available_objects = self.find_objects(game_text,cutoff=cutoff)
            self.cache_choose_object[game_text] = {}
            self.cache_choose_object[game_text][cutoff] = available_objects
            #note: when new items are added to inventory, all try_lists
            #are updated...
            self.populate_verb_try_list(current_state, available_objects, inventory_items)

        available_objects += inventory_items

        if self.verbose == True:
            print("CHOOSING AN OBJECT")
            print(self.verb_try_list[current_state])

        rand.shuffle(available_objects)
        #for o in available_objects + ['']:
        for o in available_objects:
            if len(self.verb_try_list[current_state][o]) > 0:
                #print("Selecting " + o)
                #print(self.verb_try_list[current_state][o])
                return o

        #none of the objects have actions left to explore,
        #so... increase frustration level and attempt to navigate
        print('I AM FRUSTRATED')
        self.frustration[current_state] += 1
        if self.frustration[current_state] > self.max_frustration_level:
            self.frustration[current_state] = self.max_frustration_level
        return ''

    def get_valid_transitions(self,current_state=None):
        if current_state is None:
            return self.navigation_list

        navigation_actions = []
        if current_state in valid_transitions.keys():
            #return any navigation actions we haven't tried yet
            for action in self.navigation_list:
                if action not in self.valid_transitions[current_state]:
                   navigation_actions.append(action)
            #add in any actions that we've tried, that were good
            print(self.valid_transitions[current_state])
            for action in self.valid_transitions[current_state]:
                navigation_actions.append(action)
        else:
            return list(set(self.navigation_list))

        if len(navigation_actions) == 0:
            #this should actually never happen....
            return self.navigation_list
        return list(set(navigation_actions))

    def explore(self,current_state=None):
        nav_list = self.navigation_list[:]
        
        if current_state is not None:
            if current_state not in self.frustration.keys():
                self.frustration[current_state] = 0 #frustration always defaults to 0
            frustration = self.frustration[current_state]
            if self.frustration_values[frustration]['frustration_nav_list'] == True:
                nav_list += self.frustrated_navigation    

        return rand.choice(nav_list)



    def gimme_actions(self,objects):
        try_list = []
        for o in objects:
            if len(o.split(' ')) > 0:
                base_object = o.split(' ')[-1]
            else:
                base_object = o
            if self.scholar.exists_in_model(base_object.lower()+"_NN"):
                v_aff = self.strategist.get_relation(base_object.lower()+"_NN",'affordance',num_matches=15)
                words = self.strategist.closest_words_fast(v_aff)
            else:
                words = []
            ##!pisa_words = self.strategist.closest_pisa_words(v_aff,metric='affordance')
            #print "\nWORDS FOR '" + o + "'"
            #print words
            ##!print pisa_words

            for w in self.strategist.strip_tags(words):
                if w != base_object:
                    try_list.append(w + ' ' + o)

        action_list = list(set(try_list))
        #print action_list[:100]

	action_vectors=[]
	for a in action_list:
	    action_vectors.append(self.strategist.encode(a))

	return action_list,action_vectors

    def gimme_objects(self,goal,object_list,cutoff=-1,use_pisa=False,num_matches=15,return_distance=False):
	if '' in object_list:
	    raise ValueError("object_list cannot contain empty object ''")

	if cutoff == -1:
            cutoff = self.prereq_cutoff #default is 2
	    if use_pisa==True:
		cutoff=cutoff*2

        analogy_vector = self.strategist.canonical_vectors['object_for_goal_specific']
        #analogy_vector = self.strategist.canonical_vectors['object_for_goal_test']
        v_goal = self.strategist.encode(goal) + analogy_vector

	v_objects=[]
	for o in object_list:
	    v_objects.append(self.strategist.encode(o))

        prereqs, distances = self.strategist.proximity_sort(v_goal,v_objects,object_list, cutoff=cutoff, return_distance=True, use_pisa=use_pisa, pisa_vector=analogy_vector)
	

	if return_distance == True:
	    return prereqs,distances
	else:
	    return prereqs
	
    def gimme_verbs(self,goal,verb_list,cutoff=-1,use_pisa=False,return_distance=False):
	if cutoff == -1:
            cutoff = self.prereq_cutoff #default is 2
	    if use_pisa==True:
		cutoff=cutoff*2

        analogy_vector = self.strategist.canonical_vectors['verb_for_goal']
        v_goal = self.strategist.encode(goal) + analogy_vector

	v_verbs=[]
	for v in verb_list:
	    v_verbs.append(self.strategist.encode(v))

        prereqs, distances = self.strategist.proximity_sort(v_goal,v_verbs,verb_list, cutoff=cutoff, return_distance=True, use_pisa=use_pisa, pisa_vector=analogy_vector)

	if return_distance == True:
	    return prereqs,distances
	else:
	    return prereqs
	

    def get_verbs_for_untagged_noun(self,o,search_limit=60):
        if self.scholar.exists_in_model(o.lower()+"_NN"):
            v_aff = self.strategist.get_relation(o.lower()+"_NN",'affordance',num_matches=search_limit)
            words = self.strategist.closest_words_fast(v_aff,num_matches=search_limit)
	elif self.scholar.exists_in_model(o.lower()+"_NNS"):
            v_aff = self.strategist.get_relation(o.lower()+"_NNS",'affordance',num_matches=search_limit)
            words = self.strategist.closest_words_fast(v_aff,num_matches=search_limit)
        else:
	    o_split = o.lower().split(' ')[-1]
	    try:
                v_aff = self.strategist.get_relation(o_split+'_'+self.scholar.get_most_common_tag(o_split),'affordance',num_matches=search_limit)
                words = self.strategist.closest_words_fast(v_aff,num_matches=search_limit)
	    except:
		words = []
	final_words = []
	for w in words:
	    if w[-3:] == '_VB':
		final_words.append(w)
	return final_words


    #def gimme_subgoals(self,goal,action_list,action_vectors,use_pisa=False,num_matches=15,cutoff=-1):
    def gimme_subgoals(self,goal,object_list=None,use_pisa=False,num_verbs_per_noun=5,num_nouns=5,cutoff=-1,double_filter=False,num_verbs_sampled=60,essential_verb_list = []):
	
	#words = self.scholar.get_most_common_words('VB',200)
	if object_list is None:
	    object_list = self.strategist.strip_tags(self.scholar.get_most_common_words('NN',1000))

	subgoals=[]
	objects=self.gimme_objects(goal,object_list,use_pisa=use_pisa)
	for o in objects[:num_nouns]:
	    print o
	    #get a list of affordant verbs
	    words = self.get_verbs_for_untagged_noun(o.lower())

	    untagged_words = self.strategist.strip_tags(words) + essential_verb_list
	
	    #prioritize based on relevance to goal
	    verbs = self.gimme_verbs(goal,untagged_words,use_pisa=use_pisa)
	    
	    print verbs

	    #take the top n
	    for v in verbs[:num_verbs_per_noun]:
	 	#subgoals.append(self.strategist.untag(v) + ' ' + o)
	 	subgoals.append(v + ' ' + o)

	print '\n'
	if double_filter==True:
	    #NOTE: double-filtering appears to not work very well at all...
	    #But maybe that's because I wasn't using the pisa analogy vector before...?
	    subgoal_vectors = []
	    for s in subgoals:
		subgoal_vectors.append(self.strategist.encode(s))
	    prioritized_subgoals = self.prerequisites(goal, subgoals, subgoal_vectors, use_pisa=use_pisa,cutoff=cutoff)
	    return prioritized_subgoals
	else:
	    return subgoals

    def gimme_results(self,action,use_pisa=False):
	pass

    def choose_goal(self, game_text):
        sentences = [sen.strip() for sen in game_text.split('.')]
        print(sentences)
        sentence_tags = ['ia_invalid_action','ia_navigation_failed','ia_location','ia_update_inventory','ia_environment_info']
        for sen in sentences:
            for t in sentence_tags:
                print(self.strategist.categorize(sen, t))

        raw_input('>')
        return ''


#potentially useful functions:
#find objects of interest
#determine which state is most relevant to an inventory item
#determine which state seems most 'interesting', in terms of wanting to return
#choose a goal

#interest level - limit actions spent in uninteresting states
#frustration level
#prepositions on game cues
#FOR now, prepositions only with game prompts
