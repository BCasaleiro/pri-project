from nltk.corpus import stopwords
import operator

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

def generate_graph(text, mn, mx):
    text = text.lower()
    sentences = text.split('.')
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
                e1 = edge (index, i)
                n.edges.append( e )
                nodes[index].edges.append( e1 )

    return g

def page_rank(d, n, g, m):
    print 'iteration 0'
    pr = [[(float(1) / n) for x in range(n)] for y in range(m)]
    pr_dict = dict ()

    for i in range (1, m):
        print 'iteration {}'.format(i)
        for c_index, c in enumerate(g.nodes):
            s = 0

            for e in c.edges:
                s += float(pr[i - 1][e.target]) / len(g.nodes[e.source].edges)

            pr[i][c_index] = ( (float(d) / n) + (1 - d) ) * s

    for i, p in enumerate(pr[m - 1]):
        pr_dict[g.nodes[i].ngram] = p

    top_ranked = sorted(pr_dict.items(), key=operator.itemgetter(1))
    l_t = len(top_ranked) - 1

    for i in range(l_t - 5, l_t):
        print '{} {}'.format(top_ranked[i][0], top_ranked[i][1])

def main():
    file_content = read_file('document.txt')
    print 'read file'
    g = generate_graph(file_content, 1, 3)
    print 'generated graph'
    page_rank(0.15, len(g.nodes), g, 50)
    print 'page rank'

if __name__ == '__main__':
    main()
