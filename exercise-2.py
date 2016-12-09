from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import operator
import glob

class graph(object):

    def __init__(self):
        self.nodes = []

class node(object):

    def __init__(self, ngram, position, tfidf_score):
        self.ngram = ngram
        self.counter = 1
        self.position = position
        self.tfidf_score = tfidf_score
        self.edges = []

class edge(object):

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.counter = 1

def read_file(path):
    f = open(path)
    content = f.read()
    f.close()

    return content

def calc_idf(fpath, mn, mx):
    of = []
    for f in fpath:
        aux = open(f).read()
        aux = aux.decode('utf-8','ignore').encode("utf-8")
        of.append( aux )
    vectorizer = TfidfVectorizer(strip_accents='ascii',ngram_range=(mn,mx),stop_words='english', min_df=1)
    X = vectorizer.fit_transform(of)
    idf = vectorizer.idf_
    return dict ( zip ( vectorizer.get_feature_names(), idf ) )

def calc_tf(path, mn, mx):
    f = open(path).read()
    f = f.decode('utf-8','ignore').encode("utf-8")

    vectorizer = CountVectorizer(strip_accents='ascii',ngram_range=(mn,mx),stop_words='english')
    analyze = vectorizer.build_analyzer()
    ngrams = analyze(f)

    return dict ( Counter(ngrams) )

def calc_tf_idf(tf, idf):
    tf_idf = dict ()
    for ngram in tf:
        if ngram in idf:
            tf_idf[ngram] = tf[ngram] * idf[ngram]
        else:
            tf_idf[ngram] = tf[ngram] * log(30/1)
    return tf_idf

def generate_graph(fcontent, tf, idf, tfidf, mn, mx):
    vectorizer = CountVectorizer(strip_accents='ascii',ngram_range=(mn,mx),stop_words='english')
    analyze = vectorizer.build_analyzer()
    g = graph ()

    sentences = fcontent.replace('\n', '.').replace('?','.').replace('!','.').decode('utf-8','ignore').encode("utf-8").split('.')

    ngrams = dict ()

    for sentence_index, sentence in enumerate(sentences):
        ngram_nodes = []
        sentence_ngrams = analyze(sentence)

        for ng in tf.keys():
            ng = ng.decode('utf-8','ignore').encode("utf-8")
            if ng in sentence_ngrams:
                if ng in ngrams:
                    g.nodes[ngrams[ng]].counter += 1
                    ngram_nodes.append( ngrams[ng] )
                else:
                    g.nodes.append( node (ng, sentence_index, tfidf[ng]) )
                    ngrams[ng] = len( g.nodes ) - 1
                    ngram_nodes.append( ngrams[ng] )

        for ngi, ng in enumerate(ngram_nodes):
            for ngie in range(ngi + 1, len(ngram_nodes)):

                flag = False
                for e in g.nodes[ng].edges:
                    if e.target == ngram_nodes[ngie]:
                        flag = True
                        break

                if flag:
                    e.counter += 1
                else:
                    ne = edge ( ng, ngram_nodes[ngie] )
                    ne2 = edge ( ngram_nodes[ngie], ng )
                    g.nodes[ng].edges.append( ne )
                    g.nodes[ ngram_nodes[ngie] ].edges.append( ne2 )

    return g

def page_rank(d, n, g, m):
    pr = [[(float(1) / n) for x in range(n)] for y in range(m)]
    pr_dict = dict ()

    for i in range (1, m):
        for nod_index, nod in enumerate(g.nodes):
            s = 1
            ctfidf = 1

            for e in nod.edges:
                ctfidf += g.nodes[e.target].tfidf_score

                ts = 1
                for el in g.nodes[e.target].edges:
                    ts += el.counter
                s += float( pr[i - 1][e.target] * e.counter ) / ts

            pr[i][nod_index] = ( ( float(d) * (float(nod.tfidf_score) / ctfidf) ) + (1 - d) ) * s

    for i, p in enumerate(pr[m - 1]):
        pr_dict[g.nodes[i].ngram] = p

    top_ranked = sorted(pr_dict.items(), key=operator.itemgetter(1))
    l_t = len(top_ranked)

    out = []
    for i in range(l_t - 5, l_t):
        out.append(top_ranked[i][0])

    return out

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

    debug = False

    if debug:
        print '[DEBUG] calculating idf'
    idf = calc_idf(dataset_files, 1, 3)

    for findex, fpath in enumerate(dataset_files):
        keys = read_file(key_files[findex]).split('\n')

        fcontent = read_file(fpath)

        if debug:
            print '[DEBUG] calculating tf'
        tf = calc_tf(fpath, 1, 3)

        if debug:
            print '[DEBUG] calculating tf-idf'
        tfidf = calc_tf_idf(tf, idf)

        if debug:
            print '[DEBUG] generating graph'
        g = generate_graph(fcontent, tf, idf, tfidf, 1, 3)

        if debug:
            print '[DEBUG] page rank'
        sw = page_rank(0.15, len(g.nodes), g, 50)

        if debug:
            print '[DEBUG] getting the key'
        rw = open(key_files[findex]).read().split('\n')

        if debug:
            print '[DEBUG] calculating metrics'
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

    print '[PRECISION MEAN] {}'.format(float(sum_prec)/len(dataset_files))

if __name__ == '__main__':
    main()
