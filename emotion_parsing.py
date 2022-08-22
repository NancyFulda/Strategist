import pickle

def build_from_file(filename='ijcnlp_dailydialog/emotions_correct_categories.pkl'):
  answers={}
  labels = {}
  categories=['EMO_anger','EMO_disgust','EMO_fear', 'EMO_happiness', 'EMO_sadness', 'EMO_surprise']

  labeled_sentences = pickle.load(open(filename, 'rb'))
  
  #first we need to scoop out our canonical examples
  for c in categories:
    outfile = open('canonical_examples/' + c + '.txt', 'w')
    count = 0
    for sentence, label in labeled_sentences[:]:
        if count >= 10:
            break
        if label == c:
            count+=1
            try:
                outfile.write(sentence.encode('utf8') + ' :: ' + sentence.encode('utf8') + '\n')
            except:
                outfile.write(sentence + ' :: ' + sentence + '\n')
            labeled_sentences.remove((sentence,label))

  #now put the remainder in the right format
  for entry in labeled_sentences:
    #print(entry)
    classifications = [False, False, False, False, False, False]
    for i in range(len(categories)):
        if categories[i] == entry[1]:
             classifications[i] = True
    #print(classifications)
    answers[entry[0]] = classifications
    labels[entry[0]] = entry[0]


  return answers, labels
