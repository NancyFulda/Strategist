import strategist
import numpy as np
import sys, os
from os import walk
import ms_captionbot
import time
import clarifai_labels
import skyrim_parsing
from scipy import spatial

#categorization_method = 'SKIPTHOUGHT'
#categorization_method = 'SKIPTHOUGHT_CENTROID'
categorization_method = 'SKIPTHOUGHT_NEAREST'
#categorization_method = 'SPACY_CENTROID'
#categorization_method = 'SPACY_NEAREST'
#categorization_method = 'WORD2VEC_CENTROID'

correct_categories = skyrim_parsing.build_from_file()

if categorization_method in ['SKIPTHOUGHT','SKIPTHOUGHT_CENTROID','SKIPTHOUGHT_NEAREST']:
    print "Importing Strategist..."
    s=strategist.Strategist(penseur=True,scholar=False)
elif categorization_method in ['SPACY_CENTROID','SPACY_NEAREST']:
    import spacy
    nlp = spacy.load('en')
else:
    raise ValueError('Unknown categorization method: ' + categorization_method)

categories = ['KEG_threat','KEG_explore','KEG_barter','KEG_puzzle']
#categories = ['KEG_threat','KEG_explore','KEG_barter']

baseline_words={}
baseline_words['KEG_threat'] = ['soldier','sword','badly','wounded','massive','troll','bars','bull','charges','poisinous','spider','deadly','bite','danger','fall','height','die','battle','rages','rage','angry','man','attack','plummet','plummetting','strike']
baseline_words['KEG_barter'] = ['storekeeper','shop','barter','give','offer','offers','wallet','money','marketplace','busy','bustling','selling','street','vendors','shout','hawking','hawk','purchase','wine','casks','inkeeper','merchant','order','sign','sale']
baseline_words['KEG_explore'] = ['standing','windswept','plateau','high','mountains','wall','lovely','paintings','pile','leaves','ground','window','ajar','cute','puppy','path','entryway','vaulted','beautiful','sconces','walls']
baseline_words['KEG_puzzle'] = ['treasure','chest','padlock','lever','door','locked','reagents','potion','anvil','forging','crafting','table','create','masterpiece','prison','bars','impenetrable','gate','locked','materials','butter','salt','nightshade','ginger','ginseng','garlic','wall','panel','levers','dials','puzzle','box','open']


if categorization_method in ['SKIPTHOUGHT','SKIPTHOUGHT_CENTROID','SKIPTHOUGHT_NEAREST']:
    canonical_examples = {}
    canonical_vectors = {}
    canonical_centroids = {}
    for c in categories:
        s.import_canonical_examples(c,'penseur')
        canonical_examples[c] = []
        canonical_vectors[c] = []
        for line in open('canonical_examples/' + c + '.txt', 'r').read().strip().split('\n'):
            text = line.split(' :: ')[1].strip()
            canonical_examples[c].append(text)
            canonical_vectors[c].append(s.encode(text))
        canonical_centroids[c] = np.mean(np.array(canonical_vectors[c]),0)
elif categorization_method in['SPACY_CENTROID','SPACY_NEAREST']:
    canonical_examples = {}
    canonical_vectors = {}
    canonical_centroids = {}
    for c in categories:
        canonical_examples[c] = []
        canonical_vectors[c] = []
        for line in open('canonical_examples/' + c + '.txt', 'r').read().strip().split('\n'):
            line = unicode(line)
            text = line.split(' :: ')[1].strip()
            canonical_examples[c].append(nlp(text))
            canonical_vectors[c].append(nlp(text).vector)
        canonical_centroids[c] = np.mean(np.array(canonical_vectors[c]),0)
else:
    raise ValueError('Unknown categorization method: ' + categorization_method)

print "Import complete!"

print "collecting skyrim files"
#basedir='Skyrim dataset combined small'
basedir='Skyrim dataset combined'
#basedir='Skyrim dataset images with human captions'
filenames=os.listdir(basedir)
print filenames

def categorize(text):
    if categorization_method == 'SKIPTHOUGHT':
        raise ValueError("Method SKIPTHOUGHT method no longer supported")
        return s.categorize(s.encode(text), category, mode='skipthought')
    elif categorization_method == 'SKIPTHOUGHT_CENTROID':
        doc_vector = s.encode(text)
        centroid_similarities = {}
        distances = []
        cat_vector = []
        for c in categories:
            centroid_similarities[c] = spatial.distance.cosine(doc_vector, canonical_centroids[c])
            distances.append(1-centroid_similarities[c])
            cat_vector.append((1-centroid_similarities[c]) < 0.75)
        val = max(centroid_similarities, key=centroid_similarities.get)
        return val, cat_vector, distances
    elif categorization_method == 'SKIPTHOUGHT_NEAREST':
        doc_vector = s.encode(text)
        nearest_category = None
        nearest_distance = 1000
        for c in categories:
            for e in canonical_examples[c]:
                dist = spatial.distance.cosine(doc_vector, s.encode(e))
                if dist < nearest_distance:
                    nearest_category = c
                    nearest_distance = dist
        return nearest_category, None, None
    elif categorization_method  == 'SPACY_CENTROID':
        doc_vector = nlp(unicode(text)).vector
        centroid_similarities = {}
        distances = []
        cat_vector = []
        for c in categories:
            centroid_similarities[c] = spatial.distance.cosine(doc_vector, canonical_centroids[c])
            distances.append(1-centroid_similarities[c])
            cat_vector.append((1-centroid_similarities[c]) < 0.75)
        val = max(centroid_similarities, key=centroid_similarities.get)
        return val, cat_vector, distances
    elif categorization_method == 'SPACY_NEAREST':
        doc_vector = nlp(unicode(text)).vector
        nearest_category = None
        nearest_distance = 1000
        for c in categories:
            for e in canonical_examples[c]:
                dist = spatial.distance.cosine(doc_vector, e.vector)
                if dist < nearest_distance:
                    nearest_category = c
                    nearest_distance = dist
        return nearest_category, None, None
    else:
        raise ValueError('Unknown categorization method: ' + categorization_method)

clarifai_num_correct=0
clarifai_num_false=0
clarifai_num_exact_matches=0

captionbot_num_correct=0
captionbot_num_false=0
captionbot_num_exact_matches=0

clarifai_baseline_num_correct=0
clarifai_baseline_num_false=0
clarifai_baseline_num_exact_matches=0

captionbot_baseline_num_correct=0
captionbot_baseline_num_false=0
captionbot_baseline_num_exact_matches=0

for f in filenames:

    time.sleep(0.2)

    for m in ['skipthought']:

	print 'Parsing image ' + basedir + '/' + f
	text1 = ms_captionbot.gimme_a_caption(basedir + '/' + f)
	split_text1 = text1.split(' ')
	augmented_text1 = text1 + '  ' + split_text1[0] + ' ' + split_text1[1] + ' ' + split_text1[2]
	print 'captionbot: ' + text1
        
	output_string = 'CATEGORIES: '
	    
	#captionbot
    	val, cat_vector, distances = categorize(text1)

        print val
        #print cat_vector
        #print distances
        #raw_input('pause')

        if correct_categories[f][categories.index(val)] == True:
            captionbot_num_correct += 1
        else:
            captionbot_num_false += 1

        #check for exact matches
        if cat_vector is not None:
            if cat_vector == correct_categories[f]:
                print "MATCH"
                captionbot_num_exact_matches += 1


	print output_string + ", correct answer: " + str(correct_categories[f]) + '\n'

	text2 = clarifai_labels.gimme_a_caption(basedir + '/' + f)
	print 'clarifai: ' + text2
	split_text2 = text2.split(' ')
	augmented_text2 = text2 + ' ' + split_text2[0] + ' ' + split_text2[1] + ' ' + split_text2[2]

        output_string = 'CATEGORIES: '

    	val, cat_vector, distances = categorize(text2)
        print val

        if correct_categories[f][categories.index(val)] == True:
            clarifai_num_correct += 1
        else:
            clarifai_num_false += 1

        #check for exact matches
        if cat_vector is not None:
            if cat_vector == correct_categories[f]:
                print "MATCH"
                clarifai_num_exact_matches += 1

	print output_string + ", correct answer: " + str(correct_categories[f]) +  '\n'

print '\n\n'
print "OVERALL RESULTS: \n'"

print "Clarifai:"
print "%d correct, %d false, %f percent" % (clarifai_num_correct, clarifai_num_false, 100.0 * float(clarifai_num_correct)/float(clarifai_num_correct + clarifai_num_false))
print "%d exact matches, %f percent" % (clarifai_num_exact_matches, 100.0*clarifai_num_exact_matches/len(filenames))
print'\n'

print "CaptionBot:"
print "%d correct, %d false, %f percent" % (captionbot_num_correct, captionbot_num_false, 100.0*float(captionbot_num_correct)/float(captionbot_num_correct + captionbot_num_false))
print "%d exact matches, %f percent" % (captionbot_num_exact_matches, 100.0*captionbot_num_exact_matches/len(filenames))
print'\n'

"""print "Baseline using CaptionBot text:"
print "%d correct, %d false, %f percent" % (captionbot_baseline_num_correct, captionbot_baseline_num_false, 100.0*float(captionbot_baseline_num_correct)/float(captionbot_baseline_num_correct + captionbot_baseline_num_false))
print "%d exact matches, %f percent" % (captionbot_baseline_num_exact_matches, 100.0*captionbot_baseline_num_exact_matches/len(filenames))
print'\n'

print "Baseline using Clarifai text:"
print "%d correct, %d false, %f percent" % (clarifai_baseline_num_correct, clarifai_baseline_num_false, 100.0 * float(clarifai_baseline_num_correct)/float(clarifai_baseline_num_correct + clarifai_baseline_num_false))
print "%d exact matches, %f percent" % (clarifai_baseline_num_exact_matches, 100.0*clarifai_baseline_num_exact_matches/len(filenames))
print'\n'"""
