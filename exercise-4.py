from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import xml.etree.ElementTree as ET
from collections import Counter
import operator
import glob

class article(object):
    def __init__(self, text, title, description):
        self.text = text
        self.title = title
        self.description = description

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

def calc_idf(news, mn, mx):
    of = []
    for a in news:
        of.append( a.text )
    vectorizer = TfidfVectorizer(strip_accents='ascii',ngram_range=(mn,mx),stop_words='english', min_df=1)
    X = vectorizer.fit_transform(of)
    idf = vectorizer.idf_
    return dict ( zip ( vectorizer.get_feature_names(), idf ) )

def calc_tf(art, mn, mx):
    f = art.text.decode('utf-8','ignore').encode("utf-8")

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

def generate_graph(art, tf, idf, tfidf, mn, mx):
    vectorizer = CountVectorizer(strip_accents='ascii',ngram_range=(mn,mx),stop_words='english')
    analyze = vectorizer.build_analyzer()
    g = graph ()

    sentences = art.text.replace('\n', '.').replace('?','.').replace('!','.').decode('utf-8','ignore').encode("utf-8").split('.')

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
            cpos = 1

            for e in nod.edges:
                cpos += g.nodes[e.target].position

                ts = 1
                for el in g.nodes[e.target].edges:
                    ts += el.counter
                s += float( pr[i - 1][e.target] * e.counter ) / ts

            pr[i][nod_index] = ( ( float(d) * (float(nod.position) / cpos) ) + (1 - d) ) * s

    for i, p in enumerate(pr[m - 1]):
        pr_dict[g.nodes[i].ngram] = p

    top_ranked = sorted(pr_dict.items(), key=operator.itemgetter(1))
    l_t = len(top_ranked)

    out = []
    for i in range(l_t - 5, l_t):
        out.append(top_ranked[i][0])

    return out

def main():
    debug = True

    output = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PRI</title><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous"></head><body><center><h1>The New York Times</h1></center><div class="container"><table class="table table-hover"><tr><th> Title <th><th> Description <th><th> Keyphrases </th></tr>'

    output_ending = '</table></div><script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script></body></html>'

    if debug:
        print '[DEBUG] parsing xml'

    tree = ET.parse('HomePage.xml')
    root = tree.getroot()
    channel = root.find('channel')
    news = []

    if debug:
        print '[DEBUG] indexing articles'

    for item in channel.findall('item'):
        title = item.find('title').text.encode("utf-8")
        description = item.find('description').text.encode("utf-8")
        news.append( article (title + "." + description, title, description) )

    if debug:
        print '[DEBUG] calculating idf'
    idf = calc_idf(news, 1, 3)

    for aindex, a in enumerate(news):
        if debug:
            print '[DEBUG] calculating tf'
        tf = calc_tf(a, 1, 3)

        if debug:
            print '[DEBUG] calculating tf-idf'
        tfidf = calc_tf_idf(tf, idf)

        if debug:
            print '[DEBUG] generating graph'
        g = generate_graph(a, tf, idf, tfidf, 1, 3)

        if debug:
            print '[DEBUG] page rank'
        sw = page_rank(0.15, len(g.nodes), g, 50)
        print sw

        output += '<tr><td>' + a.title + '<td><td>' + a.description + '<td><td>'

        for si,s in enumerate(sw):
            if si < len(sw) - 1:
                output += s + ', '
            else:
                output += s

        output += '</td></tr>'

    output += output_ending

    f = open('index.html', 'w')
    f.write(output)
    f.close()

if __name__ == '__main__':
    main()
