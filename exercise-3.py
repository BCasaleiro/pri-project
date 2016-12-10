#exercise-3.py
import math
import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups

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
    print("bm25: Creating bigrams")
    while((a + 1) < len(text)):
        lst = []
        lst.append(text[a])
        lst.append(text[a+1])
        bigrams.append(tuple(lst))
        a = a + 1
    a = 0
    print("bm25: Creating trigrams")
    while((a + 2) < len(text)):
        lst = []
        lst.append(text[a])
        lst.append(text[a+1])
        lst.append(text[a+2])
        trigrams.append(tuple(lst))
        a = a + 1
    print("bm25: Parsing 20newsgroups")
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
    print("bm25: Calculating avarage document Length")
    averageLength = (totalDocLength / len(newsgroups_train.data))
    k1 = 1.2
    b = 0.75
    a = 0
    numDocCount = 0
    freq = 0
    print("bm25: Counting documents")
    while(a < len(group)):
        if(group[a].count(ngram) == 1):
            numDocCount = numDocCount + 1
        a = a + 1
    print("bm25: Calculating IDF")
    if((len(group)-numDocCount+0.5)/(numDocCount+0.5) >= 0):
        IDF = math.log((len(group)-numDocCount+0.5)/(numDocCount+0.5))
    else:
        IDF = 0
    a = 0
    print("bm25: Calculating frequency")
    if(len(ngram) == 1):
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
	print("bm25: Returning")
    return (IDF * (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + b * (len(text)/averageLength))))

def pharaseness(ngram):
    i = 0
    temp = []
    while(i < len(ngram)):
        temp.append(nltk.pos_tag(ngram[i])[0][1])
        i += 1
    i = 0
    while(i < len(temp)):
        if(len(temp) == 1):
            if(temp[0][:2] == 'NN'):
                return 1
            else:
                return 0
        else:
            i = len(temp)
            if(temp[-1][:2] == 'NN'):
                a = 0
                aMax = (len(temp) - 1)
                while((a < aMax) & (temp[a] == 'JJ')):
                    a = a + 1
                while((a < aMax) & (temp[a][:2] == 'NN')):
                    a = a + 1
                if(a > 0):
                    if((a < aMax) & (temp[a-1][:2] == 'NN') & (temp[a] == 'IN')):
                        a = a + 1
                while((a < aMax) & (temp[a] == 'JJ')):
                        a = a + 1
                while((a < aMax) & (temp[a][:2] == 'NN')):
                    a = a + 1
                if( (a == aMax) & (temp[a][:2] == 'NN')):
                    return 1
                else:
                    return 0
            else:
                return 0
        i += 1

def main():
    try:
        stopset = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')
        print("Importing 20newsgroups train subset. It might take a while.")
        newsgroups_train = fetch_20newsgroups(subset='train')
        print("Importing files from SemEval2010-Maui/maui-semeval2010-train/,")
        fileName = glob.glob('SemEval2010-Maui/maui-semeval2010-train/*.txt')
        print("from NLM_500/NLM_500/documents/,")
        fileName += glob.glob('NLM_500/NLM_500/documents/*.txt')
        print("from fao780/")
        fileName += glob.glob('fao780/*.txt')
        print("and from theses100/theses80/text/")
        fileName += glob.glob('theses100/theses80/text/*.txt')
        fileContent = []
        fileContentKey = []
        i = 0
        print("and corresponding keyphrases")
        while i < len(fileName):
            aux = open(fileName[i]).read()
            fileContent.append(aux)
            keyFilename = fileName[i].replace("txt","key")
            aux = open(keyFilename).read()
            lst = []
            lst = aux.splitlines()
            fileContentKey.append(lst)	
            i += 1
        print("Importing files from wiki20/documents/")
        fileName = glob.glob('wiki20/documents/*.txt')
        i = 0
        print("and corresponding keyphrases")
        while i < len(fileName):
            aux = open(fileName[i]).read()
            fileContent.append(aux)
            keyFilename = fileName[i].replace("documents","teams/team1")
            keyFilename = keyFilename.replace("txt","key")
            aux = open(keyFilename).read()
            lst = []
            lst = aux.splitlines()
            fileContentKey.append(lst)	
            i += 1
	    ngrams = []
	    features = []
	    expected = []
        i = 0
        print("Tokenize files and remove stopwords")
        while i < len(fileContent):
            aux = []
            aux = tokenizer.tokenize(fileContent[i].lower())
            aux = [w for w in aux if not w in stopset]
            a = 0
            temp = []
            while(a < len(aux)):
                lst = []
                lst.append(aux[a])
                ngrams.append(tuple(lst))
                temp.append(1)
                if(a < 250):
                    temp.append(1)
                else:
                    temp.append(0)
                print("Obtain bm25 score for word "+str(a+1))
                temp.append(bm25(tuple(lst), aux, newsgroups_train, stopset, tokenizer))
                print("Checking pharaseness for word "+str(a+1))
                temp.append(pharaseness(tuple(lst)))
                #add more features here
                features.append(temp)
                temp = []
                if(aux[a] in fileContentKey[i]):
                    expected.append(1)
                else:
                    expected.append(0)
                if(a+1 < len(aux)):
                    lst = []
                    lst.append(aux[a])
                    lst.append(aux[a+1])
                    ngrams.append(tuple(lst))
                temp.append(1)
                if(a+1 < 250):
                    temp.append(1)
                else:
                    temp.append(0)
                print("Obtain bm25 score for bigram "+str(a+1))
                temp.append(bm25(tuple(lst), aux, newsgroups_train, stopset, tokenizer))
                print("Checking pharaseness for bigram "+str(a+1))
                temp.append(pharaseness(tuple(lst)))
                #add more features here
                features.append(temp)
                temp = []
                if(" ".join(lst) in fileContentKey[i]):
                    expected.append(1)
                else:
                    expected.append(0)
                if(a+2 < len(aux)):
                    lst = []
                    lst.append(aux[a])
                    lst.append(aux[a+1])
                    lst.append(aux[a+2])
                    ngrams.append(tuple(lst))
                temp.append(1)
                if(a+2 < 250):
                    temp.append(1)
                else:
                    temp.append(0)
                print("Obtain bm25 score for trigram "+str(a+1))
                temp.append(bm25(tuple(lst), aux, newsgroups_train, stopset, tokenizer))
                print("Checking pharaseness for trigram "+str(a+1))
                temp.append(pharaseness(tuple(lst)))
                #add more features here
                features.append(temp)
                temp = []
                if(" ".join(lst) in fileContentKey[i]):
                    expected.append(1)
                else:
                    expected.append(0)
                a += 1
            i += 1
        print("Perceptron Algorithm")
        weight = [0, 0, 0, 0]
        treshold = 0.001
        i = 0
        error = 1
        while(error >=  treshold):
            error = 0
            output = []
            while(i < len(features)):
                a = 0
                tmp = 0
                while(a < len(features[i])):
                    tmp += (features[i][a] * weight[a])
                    a += 1
                output.append(tmp)
                a = 0
                while(a < len(weight)):
                    weight[a] += ((expected[i] - output[i]) * (features[i][a]))
                    a += 1
                error += abs(expected[i] - output[i])
                i += 1
            error = float(error) / float(len(features))
        print(weight)
        print("Loading files from dataset/txt/")
        fileName = glob.glob('dataset/txt/*.txt')
        i = 0
        fileContent = []
        fileContentKey = []
        print("and corresponding keyphrases")
        while i < len(fileName):
            aux = open(fileName[i]).read()
            fileContent.append(aux)
            keyFilename = fileName[i].replace("txt","key/iic1", 1)
            keyFilename = keyFilename.replace("txt","key")
            print(keyFilename)
            aux = open(keyFilename).read()
            lst = []
            lst = aux.splitlines()
            fileContentKey.append(lst)	
            i += 1
        i = 0
        ngrams = []
        features = []
        expected = []
        while i < len(fileContent):
            aux = []
            aux = tokenizer.tokenize(fileContent[i].lower())
            aux = [w for w in aux if not w in stopset]
            a = 0
            temp = []
            while(a < len(aux)):
                lst = []
                lst.append(aux[a])
                ngrams.append(tuple(lst))
                temp.append(1)
                if(a < 250):
                    temp.append(1)
                else:
                    temp.append(0)
                print("Obtain bm25 score for word "+str(a+1))
                temp.append(bm25(tuple(lst), aux, newsgroups_train, stopset, tokenizer))
                print("Checking pharaseness for word "+str(a+1))
                temp.append(pharaseness(tuple(lst)))
                #add more features here
                features.append(temp)
                temp = []
                expected.append(i)
                if(a+1 < len(aux)):
                    lst = []
                    lst.append(aux[a])
                    lst.append(aux[a+1])
                    ngrams.append(tuple(lst))
                temp.append(1)
                if(a+1 < 250):
                    temp.append(1)
                else:
                    temp.append(0)
                print("Obtain bm25 score for bigram "+str(a+1))
                temp.append(bm25(tuple(lst), aux, newsgroups_train, stopset, tokenizer))
                print("Checking pharaseness for bigram "+str(a+1))
                temp.append(pharaseness(tuple(lst)))
                #add more features here
                features.append(temp)
                temp = []
                expected.append(i)
                if(a+2 < len(aux)):
                    lst = []
                    lst.append(aux[a])
                    lst.append(aux[a+1])
                    lst.append(aux[a+2])
                    ngrams.append(tuple(lst))
                temp.append(1)
                if(a+2 < 250):
                    temp.append(1)
                else:
                    temp.append(0)
                print("Obtain bm25 score for trigram "+str(a+1))
                temp.append(bm25(tuple(lst), aux, newsgroups_train, stopset, tokenizer))
                print("Checking pharaseness for trigram "+str(a+1))
                temp.append(pharaseness(tuple(lst)))
                #add more features here
                features.append(temp)
                temp = []
                expected.append(i)
                a += 1
            i += 1
        i = 0
        candLst = []
        tmp = []
        current = 0
        while(i < len(features)):
            if(expected[i] != current):
				candLst.append(tmp)
				tmp = []
				current = expected[i]
            a = 0
            score = 0
            while(a < len(features[i])):
                score += (features[i][a] * weight[a])
                a += 1
            tmp.append(tuple([ngrams[i],score]))
            i += 1
        candLst.append(tmp)
        z = 0
        while(z < len(candLst)):
            #create temporary Top5 with the first five different candidates...
            i = 0
            top5Temp = []
            while(len(top5Temp) < 5):
                if(top5Temp.count(candLst[z][i]) != 1):
                    top5Temp.append(candLst[z][i])
                i = i + 1

            #...and sort it (first item in the new list will have the highest score)
            i = 0
            top5 = []
            while(i < 5):
                a = 0
                t = top5Temp[0][1]
                while(a < len(top5Temp)):
                    if(top5Temp[a][1] >= t):
                        t = top5Temp[a][1]
                        p = a
                    a = a + 1
                top5.append(top5Temp[p])
                top5Temp.remove(top5Temp[p])
                i = i + 1

            #search rest of candidate list for terms with higher score than the current Top5
            while(i < len(candLst[z])):
                if((candLst[z][i][1] > top5[4][1]) & (top5.count(candLst[z][i]) != 1)):
                    top5.remove(top5[4])
                    top5.append(candLst[z][i])
                    a = 4
                    b = 3
                    while((top5[a][1] > top5[b][1]) & (b > (0-1))):
                        temp = top5[b]
                        top5[b] = top5[a]
                        top5[a] = temp
                        a = a - 1
                        b = b - 1
                i = i + 1
        
            #print Top5 keywords and their respective scores
            i = 0
            match = 0
            print("Keywords for document number "+str(z + 1)+" :\n")
            while(i < 5):
                print("Keyword number "+str(i + 1)+": "+str(top5[i][0])+" Score:"+str(top5[i][1])+'\n')
                if(" ".join(top5[i][0]) in fileContentKey[z]):
                    match += 1
                i = i + 1
            print("Precison for document number "+str(z + 1)+" : "+str(float(match)/len(fileContentKey[z]))+"\n")
            z += 1
        print("Program will exit soon.")

    except:
        return 0

if __name__ == '__main__':
	main()
