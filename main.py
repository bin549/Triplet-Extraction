from spacy.lang.en import English
import spacy


def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)


def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)


def appendChunk(original, chunk):
    return original + ' ' + chunk


def main():
    text = "London is the city of England."
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    print(document)
    sentences = [sent.string.strip() for sent in document.sents]
    nlp_model = spacy.load('en_core_web_sm')
    triples = []

    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''

    for sentence in sentences:
        tokens = nlp_model(sentence)
        for token in tokens:
            print(token.text, "->", token.dep_)
            if "punct" in token.dep_:
                continue
            if isRelationCandidate(token):
                relation = appendChunk(relation, token.lemma_)
            if isConstructionCandidate(token):
                if subjectConstruction:
                    subjectConstruction = appendChunk(
                        subjectConstruction, token.text)
                if objectConstruction:
                    objectConstruction = appendChunk(
                        objectConstruction, token.text)
            if "subj" in token.dep_:
                subject = appendChunk(subject, token.text)
                subject = appendChunk(subjectConstruction, subject)
                subjectConstruction = ''
            if "obj" in token.dep_:
                object = appendChunk(object, token.text)
                object = appendChunk(objectConstruction, object)
                objectConstruction = ''
            print(subject.strip(), ",", relation.strip(), ",", object.strip())
            return (subject.strip(), relation.strip(), object.strip())

    print(subjectConstruction)
    print(objectConstruction)
            # if isRelationCandidate(token):
            #     relation = appendChunk(relation, token.lemma_)
            #     print(relation)


        # triples.append(processSentence(sentence))



if __name__ == '__main__':
    main()
