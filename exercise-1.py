from nltk.corpus import stopwords
import operator
import glob

class graph(object):

    def __init__(self):
        self.nodes = []

class node(object):

    def __init__(self, ngram):
        self.ngram = ngram
        self.edges = []

class edge(object):

    def __init__(self, source, target):
        self.source = source
        self.target = target

def read_file(path):
    f = open(path)
    content = f.read()
    f.close()

    return content

def edge_repeated(n, e):
    for ed in n.edges:
        if e == ed.target:
            return True
    return False

def generate_graph(text, mn, mx):
    text = text.lower()
    sentences = text.replace('\n', '.').replace(',', '').replace('?','.').replace('!','.').decode('utf-8','ignore').encode("utf-8").split('.')
    stop_words = set(stopwords.words('english'))
    visited_words = {}
    visited_bigrams = {}
    visited_trigrams = {}
    g = graph()

    for sentence in sentences:
        words = sentence.split()
        nodes = []
        count = 0
        for word in words:
            if word not in stop_words:
                if word not in visited_words.keys():
                    n = node (word)
                    nodes.append( n )
                    g.nodes.append( n )
                    visited_words[word] = len(g.nodes) - 1
                else:
                    nodes.append(g.nodes[visited_words[word]])

                if count == 0:
                    bigram = word
                    trigram = word
                elif count == 1:
                    bigram += " " + word
                    if bigram not in visited_bigrams.keys():
                        n = node (bigram)
                        nodes.append( n )
                        g.nodes.append( n )
                        visited_bigrams[bigram] = len(g.nodes) - 1
                    else:
                        nodes.append(g.nodes[visited_bigrams[bigram]])
                    bigram = word

                    trigram_aux1 = word
                    trigram_aux2 = word
                    trigram += " " + word
                elif count == 2:
                    bigram += " " + word
                    if bigram not in visited_bigrams:
                        n = node (bigram)
                        nodes.append( n )
                        g.nodes.append( n )
                        visited_bigrams[bigram] = len(g.nodes) - 1
                    else:
                        nodes.append(g.nodes[visited_bigrams[bigram]])
                    bigram = word

                    trigram_aux1 = word
                    trigram_aux2 += " " + word

                    trigram += " " + word
                    if trigram not in visited_trigrams:
                        n = node (trigram)
                        nodes.append( n )
                        g.nodes.append( n )
                        visited_trigrams[trigram] = len(g.nodes) - 1
                    else:
                        nodes.append(g.nodes[visited_trigrams[trigram]])
                    trigram = trigram_aux2
                    trigram_aux2 = word
                else:
                    bigram += " " + word
                    if bigram not in visited_bigrams:
                        n = node (bigram)
                        nodes.append( n )
                        g.nodes.append( n )
                        visited_bigrams[bigram] = len(g.nodes) - 1
                    else:
                        nodes.append(g.nodes[visited_bigrams[bigram]])
                    bigram = word

                    trigram_aux1 = word
                    trigram_aux2 += " " + word

                    trigram += " " + word
                    if trigram not in visited_trigrams:
                        n = node (trigram)
                        nodes.append( n )
                        g.nodes.append( n )
                        visited_trigrams[trigram] = len(g.nodes) - 1
                    else:
                        nodes.append(g.nodes[visited_trigrams[trigram]])
                    trigram = trigram_aux2
                    trigram_aux2 = word
                    count = 1
                count += 1

        for i, n in enumerate(nodes):
            for index in range(i + 1, len(nodes)):
                e = edge (i, index)
                if not edge_repeated(n, index):
                    n.edges.append( e )
                e1 = edge (index, i)
                if not edge_repeated(nodes[index], i):
                    nodes[index].edges.append( e1 )


    return g

def page_rank(d, n, g, m):
    pr = [[(float(1) / n) for x in range(n)] for y in range(m)]
    pr_dict = dict ()
    sw = []

    for i in range (1, m):
        for c_index, c in enumerate(g.nodes):
            s = 0

            for e in c.edges:
                if len(g.nodes[e.source].edges) > 0:
                    s += float(pr[i - 1][e.target]) / len(g.nodes[e.source].edges)
                else:
                    s += float(pr[i - 1][e.target])

            pr[i][c_index] = ( (float(d) / n) + (1 - d) ) * s

    for i, p in enumerate(pr[m - 1]):
        pr_dict[g.nodes[i].ngram] = p

    top_ranked = sorted(pr_dict.items(), key=operator.itemgetter(1))
    l_t = len(top_ranked) - 1

    for i in range(l_t - 5, l_t):
        sw.append(top_ranked[i][0])
    return sw

def calc_precision(rw, sw):
    n = 0
    for word in sw:
        if word in rw:
            n += 1
    return float(n) / len(sw)

def calc_recall(rw, sw):
    n = 0
    for word in sw:
        if word in rw:
            n += 1
    return float(n)/len(rw)

def calc_f1(precision, recall):
    beta = 0.5
    if precision == 0 or recall == 0:
        return 0
    else:
        return ( (pow(beta, 2) + 1) * precision * recall ) / ((pow(beta, 2) * precision) + recall)

def main():
    dataset_files = glob.glob('dataset/txt/*.txt')
    key_files = glob.glob('dataset/key/iic1/*.key')

    sum_prec = 0

    for findex, fpath in enumerate(dataset_files):
        file_content = read_file(fpath)

        g = generate_graph(file_content, 1, 3)
        sw = page_rank(0.15, len(g.nodes), g, 50)

        rw = open(key_files[findex]).read().split('\n')

        prec = calc_precision(rw, sw)
        sum_prec += prec
        recall = calc_recall(rw, sw)
        f1 = calc_f1(prec, recall)

        print '[RESULTS] {}'.format(fpath)

        print sw
        print rw
        print '[PRECISION] {}'.format(prec)
        print '[RECALL] {}'.format(recall)
        print '[F1-SCORE] {}'.format(f1)
        print ' '

    print '[PRECISION MEAN] {}'.format(float(sum_prec)/len(dataset_files))

if __name__ == '__main__':
    main()
