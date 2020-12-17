from spacy.lang.zh import Chinese
import spacy


def main():
    text = "小龙女的男朋友是周恩来。"
    nlp = Chinese()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    print(document)
    sentences = [sent.string.strip() for sent in document.sents]
    nlp_model = spacy.load('zh_core_web_sm')
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


        # triples.append(processSentence(sentence))






if __name__ == '__main__':
    main()
