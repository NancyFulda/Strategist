import random
import sys, os
from os import walk


def build_from_file(filename='skyrim_correct_categories.txt'):
  answers={}
  categories=['threat','explore','barter','puzzle']

  with open(filename) as f:
    for line in f:
        groups = line.split('\t')
        print groups
        classifications=[False, False, False, False]
        for i in range(len(categories)):
	    if categories[i] in groups[1]:
	        classifications[i] = True
        print classifications
        print '\n'
        answers[groups[0]] = classifications

    print '\n'

  return answers

def get_human_labels(filename='skyrim_human_captions.txt'):
  labels={}

  with open(filename) as f:
    for line in f:
        groups = line.split('\t')
        print groups
	print '\n'
        labels['ScreenShot' + groups[0] + '.jpg'] = groups[1]

    print '\n'

  return labels 

def get_baselines(image_filenames=[],source_filename='skyrim_correct_categories.txt'):

    correct_answers = build_from_file(source_filename)

    if image_filenames == []:
	#basedir='../../Desktop/Skyrim dataset combined small'
	basedir='Skyrim dataset combined'
	image_filenames = os.listdir(basedir)

    #optimist - always thinks all categories are active
    optimist_num_correct=0
    optimist_num_false = 0
    optimist_num_exact_matches = 0
    for f in image_filenames:
	cats = [1,1,1,1]
	for i in range(len(cats)):
	    if cats[i] == correct_answers[f][i]:
	        optimist_num_correct += 1
	    else:
	        optimist_num_false += 1
	if cats == correct_answers[f]:
	    optimist_num_exact_matches += 1

    #pessimist - marks everything as false
    pessimist_num_correct=0
    pessimist_num_false = 0
    pessimist_num_exact_matches = 0
    for f in image_filenames:
	cats = [0,0,0,0]
	for i in range(len(cats)):
	    if cats[i] == correct_answers[f][i]:
	        pessimist_num_correct += 1
	    else:
	        pessimist_num_false += 1
	if cats == correct_answers[f]:
	    pessimist_num_exact_matches += 1

    #explorer - uses the most common categorization - explore
    explorer_num_correct=0
    explorer_num_false = 0
    explorer_num_exact_matches = 0
    for f in image_filenames:
	cats = [0,1,0,0]
	for i in range(len(cats)):
	    if cats[i] == correct_answers[f][i]:
	        explorer_num_correct += 1
	    else:
	        explorer_num_false += 1
	if cats == correct_answers[f]:
	    explorer_num_exact_matches += 1

    #random - just picks 1 of the categories randomly
    random_num_correct=0
    random_num_false = 0
    random_num_exact_matches = 0
    for f in image_filenames:
	cats = [0,0,0,0]
        cats[random.randint(0,3)] = 1
	for i in range(len(cats)):
	    #OLD METHOD
            #if cats[i] == correct_answers[f][i]:
	    #    random_num_correct += 1
	    #else:
	    #    random_num_false += 1
	    if correct_answers[f][cats.index(1) == 1]:
	        random_num_correct += 1
	    else:
	        random_num_false += 1
	if cats == correct_answers[f]:
	    random_num_exact_matches += 1

    print "Optimist:"
    print "%d correct, %d false, %f percent" % (optimist_num_correct, optimist_num_false, 100.0 * float(optimist_num_correct)/float(optimist_num_correct + optimist_num_false))
    print "%d exact matches, %f percent" % (optimist_num_exact_matches, 100.0*optimist_num_exact_matches/len(image_filenames))
    print'\n'

    print "Pessimist:"
    print "%d correct, %d false, %f percent" % (pessimist_num_correct, pessimist_num_false, 100.0 * float(pessimist_num_correct)/float(pessimist_num_correct + pessimist_num_false))
    print "%d exact matches, %f percent" % (pessimist_num_exact_matches, 100.0*pessimist_num_exact_matches/len(image_filenames))
    print'\n'

    print "Explorer:"
    print "%d correct, %d false, %f percent" % (explorer_num_correct, explorer_num_false, 100.0 * float(explorer_num_correct)/float(explorer_num_correct + explorer_num_false))
    print "%d exact matches, %f percent" % (explorer_num_exact_matches, 100.0*explorer_num_exact_matches/len(image_filenames))
    print'\n'

    print "Random:"
    print "%d correct, %d false, %f percent" % (random_num_correct, random_num_false, 100.0 * float(random_num_correct)/float(random_num_correct + random_num_false))
    print "%d exact matches, %f percent" % (random_num_exact_matches, 100.0*random_num_exact_matches/len(image_filenames))
    print'\n'

get_baselines()
