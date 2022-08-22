import strategist
import numpy as np
import sys, os
from os import walk
#import skyrim_parsing
import sentiment_parsing
from scipy import spatial
import embedding_apis as embed_api
import random

#Original KEG file modified for IJCAI 2019...

#correct_categories = skyrim_parsing.build_from_file()
#human_captions = skyrim_parsing.get_human_labels()

#'human captions' is actually now just the text to be analyzed
correct_categories, human_captions = sentiment_parsing.build_from_file()


categories = ['neutral','objective','positive', 'negative','objective-OR-neutral']
#categories = ['KEG_threat','KEG_explore','KEG_barter']

#baseline_words={}
#baseline_words['EMO_anger'] = ['angry','frustrated','hate','can\'t','won\'t','stop']
#baseline_words['EMO_disgust'] = ['ugh','gross','disgusting']
#baseline_words['EMO_fear'] = ['np','wait','afraid']
#baseline_words['EMO_happiness'] = ['happy','great','thank','glad','nice']
#baseline_words['EMO_sadness'] = ['sad','sorry']
#baseline_words['EMO_surprise'] = ['why','understand','don\'t']

#categorization_method = 'SKIPTHOUGHT'
#categorization_method = 'SKIPTHOUGHT_CENTROID'
#categorization_method = 'SKIPTHOUGHT_NEAREST'
#categorization_method = 'SPACY_CENTROID'
#categorization_method = 'SPACY_NEAREST'
#categorization_method = 'WORD2VEC_CENTROID'
#categorization_method = 'WORD2VEC_NEAREST'
#categorization_method = 'USE_LITE_CENTROID'
categorization_method = 'USE_LITE_NEAREST'
#categorization_method = 'USE_LARGE_CENTROID'
#categorization_method = 'USE_LARGE_NEAREST'
#categorization_method = 'BAG_OF_WORDS_CENTROID'
#categorization_method = 'BAG_OF_WORDS_NEAREST'

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
            use_large_vectors = embed_api.embed(chunk, 'use_large')
        else:
            use_large_vectors = np.vstack([use_large_vectors,embed_api.embed(chunk, 'use_large')])
        print(use_large_vectors.shape)
    for i in range(len(sentence_list)):
        pre_embedded_vectors[sentence_list[i]] = use_large_vectors[i]

    print("Done")

print("Categorization method is ", categorization_method)
print("creating canonical examples")
if categorization_method in ['SKIPTHOUGHT','SKIPTHOUGHT_CENTROID','SKIPTHOUGHT_NEAREST']:
    print "Importing Strategist..."
    s=strategist.Strategist(penseur=True,scholar=False)
elif categorization_method in ['SPACY_CENTROID','SPACY_NEAREST']:
    import spacy
    nlp = spacy.load('en')
elif categorization_method in ['USE_LITE_CENTROID','USE_LITE_NEAREST','USE_LARGE_CENTROID','USE_LARGE_NEAREST','BAG_OF_WORDS_CENTROID','BAG_OF_WORDS_NEAREST']:
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
elif categorization_method in['SPACY_CENTROID','SPACY_NEAREST','USE_LITE_CENTROID','USE_LITE_NEAREST','USE_LARGE_NEAREST', 'USE_LARGE_CENTROID','BAG_OF_WORDS_CENTROID','BAG_OF_WORDS_NEAREST']:
    canonical_examples = {}
    canonical_vectors = {}
    canonical_centroids = {}
    for c in categories:
        canonical_examples[c] = []
        canonical_vectors[c] = []
        use_large_sentences = []
        for line in open('canonical_examples/' + c + '.txt', 'r').read().strip().split('\n'):
            line = unicode(line.decode('utf8'))
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
            #canonical_examples[c].append(nlp(text))
            canonical_examples[c].append(text)
            #canonical_vectors[c].append(nlp(text).vector)
            canonical_vectors[c].append(text_vector)
        if categorization_method in ['USE_LARGE_CENTROID','USE_LARGE_NEAREST']:
            canonical_vectors[c] = embed_api.embed(use_large_sentences, 'use_large')
        canonical_centroids[c] = np.mean(np.array(canonical_vectors[c]),0)
else:
    raise ValueError('Unknown categorization method: ' + categorization_method)

#print "Import complete!"
print("done")

def categorize(text):
    if categorization_method == 'SKIPTHOUGHT':
        raise ValueError("Method SKIPTHOUGHT method no longer supported")
        return s.categorize(s.encode(text), category, mode='skipthought')
    elif categorization_method == 'SKIPTHOUGHT_CENTROID':
        doc_vector = s.encode(text.decode('utf8'))
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
        doc_vector = s.encode(text.decode('utf8'))
        nearest_category = None
        nearest_distance = 1000
        for c in categories:
            for e in canonical_examples[c]:
                dist = spatial.distance.cosine(doc_vector, s.encode(e))
                if dist < nearest_distance:
                    nearest_category = c
                    nearest_distance = dist
        cat_vector=[0,0,0,0,0]
        cat_vector[categories.index(nearest_category)] = True
        return nearest_category, cat_vector, None
    elif categorization_method  in ['SPACY_CENTROID', 'USE_LITE_CENTROID', 'USE_LARGE_CENTROID','BAG_OF_WORDS_CENTROID']:
        if categorization_method == 'SPACY_CENTROID':
            doc_vector = nlp(unicode(text.decode('utf8'))).vector
        elif categorization_method == 'USE_LITE_CENTROID':
            doc_vector = embed_api.embed([text], 'use_lite')[0]
        elif categorization_method == 'USE_LARGE_CENTROID':
            #doc_vector = embed_api.embed([text], 'use_large')[0]
            doc_vector = pre_embedded_vectors[text]
        elif categorization_method == 'BAG_OF_WORDS_CENTROID':
            doc_vector = embed_api.embed([text], 'bag_of_words')[0]
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
    elif categorization_method in ['SPACY_NEAREST', 'USE_LITE_NEAREST','USE_LARGE_NEAREST','BAG_OF_WORDS_NEAREST']:
        if categorization_method == 'SPACY_NEAREST':
            doc_vector = nlp(unicode(text.decode('utf8'))).vector
        elif categorization_method == 'USE_LITE_NEAREST':
            doc_vector = embed_api.embed([text], 'use_lite')[0]
        elif categorization_method == 'USE_LARGE_NEAREST':
            #doc_vector = embed_api.embed([text], 'use_large')[0]
            doc_vector = pre_embedded_vectors[text]
        elif categorization_method == 'BAG_OF_WORDS_NEAREST':
            doc_vector = embed_api.embed([text], 'bag_of_words')[0]
        nearest_category = None
        nearest_distance = 1000
        for c in categories:
            for v in canonical_vectors[c]:
                #dist = spatial.distance.cosine(doc_vector, v)
                #dist = spatial.distance.cityblock(doc_vector, v)
                dist = spatial.distance.euclidean(doc_vector, v)
                if dist < nearest_distance:
                    nearest_category = c
                    nearest_distance = dist
        cat_vector=[0,0,0,0,0]
        cat_vector[categories.index(nearest_category)] = True
        return nearest_category, cat_vector, None
    else:
        raise ValueError('Unknown categorization method: ' + categorization_method)

print "collecting skyrim files"
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

false_positives = [0,0,0,0,0]
failed_detections = [0,0,0,0,0]

baseline_false_positives = [0,0,0,0,0]
baseline_failed_detections = [0,0,0,0,0]

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

        human_boolean_cats=[0,0,0,0,0]
        human_baseline_categories=[0,0,0,0,0]
        for i in range(len(categories)):
            #for word in baseline_words[categories[i]]:
            #    if word in text.split(' '):
            #        human_baseline_categories[i] = True
            
            #when you want a random baseline...
            if random.randint(0,1) == 0:
                human_baseline_categories[i] = True
            else:
                human_baseline_categories[i] = False

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

print '\n\n'
print "OVERALL RESULTS: \n'"

print "Human:"
print "%d correct, %d false, %f percent" % (human_num_correct, human_num_false, 100.0 * float(human_num_correct)/float(human_num_correct + human_num_false))
print "%d exact matches, %f percent" % (human_num_exact_matches, 100.0*human_num_exact_matches/len(human_captions.keys()))
print'\n'

print "Random:"
print "%d correct, %d false, %f percent" % (human_baseline_num_correct, human_baseline_num_false, 100.0 * float(human_baseline_num_correct)/float(human_baseline_num_correct + human_baseline_num_false))
print "%d exact matches, %f percent" % (human_baseline_num_exact_matches, 100.0*human_baseline_num_exact_matches/len(human_captions.keys()))
print'\n'

print "False positives:"
print false_positives
print "Failed detections:"
print failed_detections
print '\n'

print "Random False positives:"
print baseline_false_positives
print "Random Failed detections:"
print baseline_failed_detections
print '\n'
