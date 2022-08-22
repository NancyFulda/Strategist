import pickle

def build_from_file(filename='semeval/dist.tweets.txt', three_categories=False):
  answers={}
  labels = {}
  if three_categories:
    categories=['neutral', 'positive', 'negative']
  else:
    categories=['neutral', 'objective', 'objective-OR-neutral', 'positive', 'negative']

  labeled_sentences = open(filename, 'r').readlines()
  
  #first we need to scoop out our canonical examples
  for c in categories:
    outfile = open('canonical_examples/' + c + '.txt', 'w')
    count = 0
    for line in labeled_sentences[:]:
        _, _, label, sentence = line.split('\t')
        label = label.strip('"')
        if three_categories:
            if label in ['neutral','objective','objective-OR-neutral']:
                label = 'neutral'
        sentence = sentence.strip()
        if count >= 10:
            break
        if label == c:
            count+=1
            #outfile.write(sentence.encode('utf8') + ' :: ' + sentence.encode('utf8') + '\n')
            outfile.write(sentence + ' :: ' + sentence + '\n')
            labeled_sentences.remove(line)

  #now put the remainder in the right format
  for line in labeled_sentences:
    _, _, label, sentence = line.split('\t')
    label = label.strip('"')
    if three_categories:
        if label in ['neutral','objective','objective-OR-neutral']:
            label = 'neutral'
    entry = (sentence, label)
    #print(entry)
    if three_categories:
        classifications = [False, False, False]
    else:
        classifications = [False, False, False, False, False]
    for i in range(len(categories)):
        if categories[i] == entry[1]:
             classifications[i] = True
    #print(classifications)
    answers[entry[0]] = classifications
    labels[entry[0]] = entry[0]


  return answers, labels

if __name__ == '__main__':
  build_from_file()
