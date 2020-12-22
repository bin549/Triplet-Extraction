from spacy.lang.en import English
import spacy
import networkx as nx
import matplotlib.pyplot as plt


def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)


def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)


def appendChunk(original, chunk):
    return original + ' ' + chunk


def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    for token in tokens:
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
    return (subject.strip(), relation.strip(), object.strip())


def processSentence(sentence):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)


def printGraph(triples):
    G = nx.Graph()
    for triple in triples:
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='seagreen', alpha=0.9,
            labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    text = "London is the city of England."
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    sentences = [sent.string.strip() for sent in document.sents]
    nlp_model = spacy.load('en_core_web_sm')
    triples = []

    for sentence in sentences:
        tokens = nlp_model(sentence)
        triples.append(processSentence(sentence))

    print(triples)
    printGraph(triples)


    # return (subject.strip(), relation.strip(), object.strip())

            #     if subjectConstruction:
            #         subjectConstruction = appendChunk(subjectConstruction, token.text)
            #         print(subjectConstruction)
            #     if objectConstruction:
            #         objectConstruction = appendChunk(
            #             objectConstruction, token.text)

            #     subject = appendChunk(subjectConstruction, subject)
            #     subjectConstruction = ''

            #     object = appendChunk(objectConstruction, object)
            #     objectConstruction = ''

    # print(subjectConstruction)
    # print(objectConstruction)

        # triples.append(processSentence(sentence))
