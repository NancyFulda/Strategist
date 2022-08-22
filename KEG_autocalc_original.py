import strategist
import numpy as np
import sys, os
from os import walk
import ms_captionbot
import time
import clarifai_labels
import skyrim_parsing

correct_categories = skyrim_parsing.build_from_file()

print "Importing Strategist..."
s=strategist.Strategist(penseur=True,scholar=False)

categories = ['KEG_threat','KEG_explore','KEG_barter','KEG_puzzle']
#categories = ['KEG_threat','KEG_explore','KEG_barter']

baseline_words={}
baseline_words['KEG_threat'] = ['soldier','sword','badly','wounded','massive','troll','bars','bull','charges','poisinous','spider','deadly','bite','danger','fall','height','die','battle','rages','rage','angry','man','attack','plummet','plummetting','strike']
baseline_words['KEG_barter'] = ['storekeeper','shop','barter','give','offer','offers','wallet','money','marketplace','busy','bustling','selling','street','vendors','shout','hawking','hawk','purchase','wine','casks','inkeeper','merchant','order','sign','sale']
baseline_words['KEG_explore'] = ['standing','windswept','plateau','high','mountains','wall','lovely','paintings','pile','leaves','ground','window','ajar','cute','puppy','path','entryway','vaulted','beautiful','sconces','walls']
baseline_words['KEG_puzzle'] = ['treasure','chest','padlock','lever','door','locked','reagents','potion','anvil','forging','crafting','table','create','masterpiece','prison','bars','impenetrable','gate','locked','materials','butter','salt','nightshade','ginger','ginseng','garlic','wall','panel','levers','dials','puzzle','box','open']


for c in categories:
    s.import_canonical_examples(c,'penseur')

print "Import complete!"

print "collecting skyrim files"
#basedir='Skyrim dataset combined small'
basedir='Skyrim dataset combined'
#basedir='Skyrim dataset images with human captions'
filenames=os.listdir(basedir)
print filenames


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
	print 'augmented captionbot: ' + augmented_text1
        v_text1 = s.encode(text1, mode=m)
        augmented_v_text1 = s.encode(augmented_text1, mode=m)
        
	output_string = 'CATEGORIES: '
	captionbot_boolean_cats=[0,0,0,0]
	captionbot_baseline_categories=[0,0,0,0]
        for i in range(len(categories)):
	    
	    #captionbot
	    val = s.categorize(v_text1, categories[i], mode=m)
	    #val2 = s.categorize(augmented_v_text1, categories[i], mode=m)

	    #val = False
	    #if val1 == True or val2 == True:
	    #	val = True

	    if val == correct_categories[f][i]:
		captionbot_num_correct += 1
	    else:
		captionbot_num_false += 1

	    if val == True:
		captionbot_boolean_cats[i] = True
		output_string += categories[i] + ', '

	    #baseline using captionbot text
	    for word in baseline_words[categories[i]]:
		if word in text1.split(' '):
		    captionbot_baseline_categories[i] = True

	    if captionbot_baseline_categories[i] == correct_categories[f][i]:
		captionbot_baseline_num_correct += 1
	    else:
		captionbot_baseline_num_false += 1

	#check for exact matches
	if captionbot_boolean_cats == correct_categories[f]:
	    print "MATCH"
	    captionbot_num_exact_matches += 1
	if captionbot_baseline_categories == correct_categories[f]:
	    captionbot_baseline_num_exact_matches += 1


	print output_string + ", correct answer: " + str(correct_categories[f]) + '\n'
	print 'baseline with captionbot:' + str(captionbot_baseline_categories)

	text2 = clarifai_labels.gimme_a_caption(basedir + '/' + f)
	print 'clarifai: ' + text2
	split_text2 = text2.split(' ')
	augmented_text2 = text2 + ' ' + split_text2[0] + ' ' + split_text2[1] + ' ' + split_text2[2]
	print 'augmented clarifai text: ' + augmented_text2
        v_text2 = s.encode(text2, mode=m)
        augmented_v_text2 = s.encode(augmented_text2, mode=m)

        output_string = 'CATEGORIES: '
	clarifai_boolean_cats=[0,0,0,0]
	clarifai_baseline_categories=[0,0,0,0]
        for i in range(len(categories)):

	    #clarifai
	    val = s.categorize(v_text2, categories[i], mode=m)
	    #val2 = s.categorize(augmented_v_text2, categories[i], mode=m)

	    #val = False
	    #if val1 == True or val2 == True:
	    #	val = True

	    if val == correct_categories[f][i]:
		clarifai_num_correct += 1
	    else:
		clarifai_num_false += 1

	    if val == True:
		clarifai_boolean_cats[i] = True
		output_string += categories[i] + ', '

	    #baseline using clarifai txt
	    for word in baseline_words[categories[i]]:
		if word in text2.split(' '):
		    clarifai_baseline_categories[i] = True

	    if clarifai_baseline_categories[i] == correct_categories[f][i]:
		clarifai_baseline_num_correct += 1
	    else:
		clarifai_baseline_num_false += 1

	#check perfect matches
	if clarifai_boolean_cats == correct_categories[f]:
	    print "MATCH"
	    clarifai_num_exact_matches += 1
	if clarifai_baseline_categories == correct_categories[f]:
	    clarifai_baseline_num_exact_matches += 1

	print output_string + ", correct answer: " + str(correct_categories[f]) +  '\n'
	print 'baseline with clarifai:' + str(clarifai_baseline_categories)

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

print "Baseline using CaptionBot text:"
print "%d correct, %d false, %f percent" % (captionbot_baseline_num_correct, captionbot_baseline_num_false, 100.0*float(captionbot_baseline_num_correct)/float(captionbot_baseline_num_correct + captionbot_baseline_num_false))
print "%d exact matches, %f percent" % (captionbot_baseline_num_exact_matches, 100.0*captionbot_baseline_num_exact_matches/len(filenames))
print'\n'

print "Baseline using Clarifai text:"
print "%d correct, %d false, %f percent" % (clarifai_baseline_num_correct, clarifai_baseline_num_false, 100.0 * float(clarifai_baseline_num_correct)/float(clarifai_baseline_num_correct + clarifai_baseline_num_false))
print "%d exact matches, %f percent" % (clarifai_baseline_num_exact_matches, 100.0*clarifai_baseline_num_exact_matches/len(filenames))
print'\n'
