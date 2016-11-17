from nltk.corpus import stopwords

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
                n.edges.append( e )
                nodes[index].edges.append( e )

    return g

def main():
    file_content = read_file('document.txt')
    g = generate_graph(file_content, 1, 3)
    for n in g.nodes:
        print '[{}]\t{}'.format(len(n.edges), n.ngram)

if __name__ == '__main__':
    main()
