import strategist
import numpy as np


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

print "Enter the text to be categorized"
text = raw_input()
while text != 'exit' and text != 'quit' and text != 'done':
    if text == '':
        text = raw_input()
	continue

    #for m in ['skipthought','uniskip','biskip','reverse','forward','treverse','tforward','mini']:
    for m in ['skipthought', 'forward', 'reverse']:
        v_text = s.encode(text, mode=m)
	print '\nMODE: ' + m  + ' ' + str(len(v_text))
	print np.mean(v_text)

        output_string = ''
        for c in categories:

	    val = s.categorize(v_text, c, mode=m)
	    if val == True:
		output_string += c + ', '

	    baseline = False
	    for word in baseline_words[c]:
	        if word in text.split(' '):
		    baseline=True	

	    #print c + ': ' +  str(val) + ' (' + str(baseline) + ')'
	print output_string

    text = raw_input("\n Next caption: ")
