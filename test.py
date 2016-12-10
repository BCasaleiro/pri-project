import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups
import logging

def bm25(ngram, text, newsgroups_train, stopset, tokenizer):
	#create bigrams and trigrams for text
	#add bigrams and trigrams to each list representing a document
	#also calculate average document length
	group = []
	i = 0
	totalDocLength = 0
	tempLst = []
	bigrams = []
	trigrams = []
	a = 0
	while((a + 1) < len(text)):
		lst = []
		lst.append(text[a])
		lst.append(text[a+1])
		bigrams.append(tuple(lst))
		a = a + 1
	a = 0
	while((a + 2) < len(text)):
		lst = []
		lst.append(text[a])
		lst.append(text[a+1])
		lst.append(text[a+2])
		trigrams.append(tuple(lst))
		a = a + 1
	while(i < len(newsgroups_train.data)):
		temp = tokenizer.tokenize(newsgroups_train.data[i])
		temp = [w for w in temp if not w in stopset]
		totalDocLength = totalDocLength + len(temp)
		a = 0
		while((a + 1) < len(temp)):
			lst = []
			lst.append(temp[a])
			lst.append(temp[a+1])
			tempLst.append(tuple(lst))
			a = a + 1
		a = 0
		while((a + 2) < len(temp)):
			lst = []
			lst.append(temp[a])
			lst.append(temp[a+1])
			lst.append(temp[a+2])
			tempLst.append(tuple(lst))
			a = a + 1
		temp.append(tempLst)
		group.append(temp)
		i = i + 1
	averageLength = (totalDocLength / len(newsgroups_train.data))
	k1 = 1.2
	b = 0.75
	a = 0
	numDocCount = 0
	freq = 0
	while(a < len(group)):
		if(group[a].count(ngram) == 1):
			numDocCount = numDocCount + 1
		a = a + 1
	if((len(group)-numDocCount+0.5)/(numDocCount+0.5) >= 0):
		IDF = math.log((len(group)-numDocCount+0.5)/(numDocCount+0.5))
	else:
		IDF = 0
	a = 0
	if(len(ngra) == 1):
		while(a < len(text)):
			if(text[a] == ngram[0]):
				freq = freq + 1
			a = a + 1
		frequency = (float(freq) / float(len(text)))
	else:
		if(len(ngram) == 2):
			while(a < len(bigrams)):
				if(bigrams[a] == ngram):
					freq = freq + 1
				a = a + 1
			frequency = (float(freq) / float(len(bigrams)))
		else:
			while(a < len(trigrams)):
				if(trigrams[a] == ngram):
					freq = freq + 1
				a = a + 1
			frequency = (float(freq) / float(len(trigrams)))
	return (IDF * (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + b * (len(text)/averageLength))))

def main():
	try:
		logging.basicConfig()
		stopset = set(stopwords.words('english'))
		tokenizer = RegexpTokenizer(r'\w+')
		print("Importing 20newsgroups train subset. It might take a while.")
		newsgroups_train = fetch_20newsgroups(subset='train')
		a = bm25(tuple(["hello"]), tokenizer.tokenize("hello world i am here"), newsgroups_train, stopset, tokenizer)
		print(a)
	except:
		return 0

if __name__ == '__main__':
	main()
