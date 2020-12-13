import spacy
from spacy import displacy
import nltk
from nltk.stem.porter import *
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.pipeline import SentenceSegmenter
from . import game


def test01():
    nlp = spacy.load('en_core_web_sm')
    # nlp = spacy.load('zh_core_web_sm')

    doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')
    doc2 = nlp(u"Tesla isn't  looking into startups anymore.")
    doc3 = nlp(u'Although commmonly attributed to John Lennon from his song "Beautiful Boy", \
        the phrase "Life is what happens to us while we are making other plans" was written by \
        cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17.')
    doc4 = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
    life_quote = doc3[16:30]

    print(spacy.explain('PROPN'))
    print(spacy.explain('nsubj'))
    print(doc[5].text + ' : ' + doc[5].shape_)
    print(nlp.pipeline)
    print(nlp.pipe_names)
    print(doc2[4].text)
    print(doc2[4].lemma_)
    print(doc2[4].pos_)
    print(doc2[4].tag_ + ' / ' + spacy.explain(doc2[4].tag_))
    print(doc2[0].text + ': ' + doc2[0].shape_)
    print(doc2[0].is_alpha)
    print(doc2[0].is_stop)
    print(life_quote)
    print(type(life_quote))
    print(doc2)
    print(doc2[0])
    print(type(doc2))
    print(doc2[0].pos_)
    print(doc2[0].dep_)
    print(doc4[6].is_sent_start)

    for token in doc:
        print(token.text, token.pos_, token.dep_)
    for token in doc2:
        print(token.text, token.pos_, token.dep_)
    for sent in doc4.sents:
        print(sent)


def test02():
    nlp = spacy.load('en_core_web_sm')
    mystring = '"We\'re moving to L.A.!"'
    doc = nlp(mystring)
    doc2 = nlp(u"We're here to help! Send snail-mail, email support@oursite.com or visit us at http://www.oursite.com!")
    doc3 = nlp(u'A 5km NYC cab ride costs $10.30')
    doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")
    doc5 = nlp(u'It is better to give than to receive.')
    doc6 = nlp(u'My dinner was horrible.')
    doc7 = nlp(u'Your dinner was delicious.')
    doc8 = nlp(u'Apple to build a Hong Kong factory for $6 million')
    doc9 = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")
    doc10 = nlp(u"Red cars do not carry higher insurance rates.")
    doc11 = nlp(u"He was a one-eyed, one-horned, flying, purple people-eater.")

    print(len(doc))
    print(len(doc.vocab))
    print(doc5[2])
    print(doc5[2:5])
    print(doc5[-4:])
    # doc6[3] = doc7[3]
    print('\n----')
    print(len(doc8.ents))

    for token in doc:
        print(token.text, end=' | ')
    for t in doc2:
        print(t)
    for t in doc3:
        print(t)
    for t in doc4:
        print(t)
    for token in doc8:
        print(token.text, end=' | ')
    for ent in doc8.ents:
        print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
    for chunk in doc9.noun_chunks:
        print(chunk.text)
    for chunk in doc11.noun_chunks:
        print(chunk.text)
    for chunk in doc10.noun_chunks:
        print(chunk.text)


def test03():
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(u'Apple is going to build a U.K. factory for $6 million.')
    displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})
    doc = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.')
    displacy.render(doc, style='ent', jupyter=True)
    doc = nlp(u'This is a sentence.')
    displacy.serve(doc, style='dep')


def test04():
    p_stemmer = PorterStemmer()
    words = ['run', 'runner', 'running', 'ran', 'runs', 'easily', 'fairly']
    # words = ['generous','generation','generously','generate']
    # words = ['consolingly']
    for word in words:
        print(word + ' --> ' + p_stemmer.stem(word))


def test05():
    nlp = spacy.load('en_core_web_sm')
    doc1 = nlp(u"I am a runner running in a race because I love to run since I ran today")
    doc2 = nlp(u"I saw eighteen mice today!")
    doc3 = nlp(u"I am meeting him tomorrow at the meeting.")
    doc4 = nlp(u"That's an enormous automobile")

    for token in doc1:
        print(token.text, '\t', token.pos_, '\t', token.lemma, '\t', token.lemma_)
    show_lemmas(doc2)
    show_lemmas(doc3)
    show_lemmas(doc4)


def test06():
    nlp = spacy.load('en_core_web_sm')
    print(nlp.Defaults.stop_words)
    print(len(nlp.Defaults.stop_words))
    print(nlp.vocab['myself'].is_stop)
    print(nlp.vocab['mystery'].is_stop)

    nlp.Defaults.stop_words.add('btw')
    nlp.vocab['btw'].is_stop = True
    print(len(nlp.Defaults.stop_words))
    print(nlp.vocab['btw'].is_stop)

    nlp.Defaults.stop_words.remove('beyond')
    nlp.vocab['beyond'].is_stop = False

    print(len(nlp.Defaults.stop_words))
    print(nlp.vocab['beyond'].is_stop)


def test07():
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    print(matcher)
    pattern1 = [{'LOWER': 'solarpower'}]
    pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
    pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
    matcher.add('SolarPower', None, pattern1, pattern2, pattern3)

    doc = nlp(
        u'The Solar Power industry continues to grow as demand for solarpower increases. Solar-power cars are gaining popularity.')
    found_matches = matcher(doc)
    print(found_matches)
    for match_id, start, end in found_matches:
        string_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        print(match_id, string_id, start, end, span.text)
    pattern1 = [{'LOWER': 'solarpower'}]
    pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP': '*'}, {'LOWER': 'power'}]
    matcher.remove('SolarPower')
    matcher.add('SolarPower', None, pattern1, pattern2)
    found_matches = matcher(doc)
    print(found_matches)
    pattern1 = [{'LOWER': 'solarpower'}]
    pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP': '*'}, {'LEMMA': 'power'}]  # CHANGE THIS PATTERN
    matcher.remove('SolarPower')
    matcher.add('SolarPower', None, pattern1, pattern2)

    doc2 = nlp(u'Solar-powered energy runs solar-powered cars.')
    found_matches = matcher(doc2)
    print(found_matches)
    pattern1 = [{'LOWER': 'solarpower'}]
    pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP': '*'}, {'LOWER': 'power'}]
    pattern3 = [{'LOWER': 'solarpowered'}]
    pattern4 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP': '*'}, {'LOWER': 'powered'}]
    matcher.remove('SolarPower')
    matcher.add('SolarPower', None, pattern1, pattern2, pattern3, pattern4)
    found_matches = matcher(doc2)
    print(found_matches)


def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')


def test08():
    nlp = spacy.load('en_core_web_sm')

    matcher = PhraseMatcher(nlp.vocab)
    with open('./TextFiles/reaganomics.txt', encoding='utf8') as f:
        doc3 = nlp(f.read())
    phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']
    phrase_patterns = [nlp(text) for text in phrase_list]
    matcher.add('VoodooEconomics', None, *phrase_patterns)
    matches = matcher(doc3)
    sents = [sent for sent in doc3.sents]

    print(matches)
    print(doc3[:70])
    print(doc3[665:685])
    print(doc3[2975:2995])
    print(sents[0].start, sents[0].end)
    for sent in sents:
        if matches[4][1] < sent.end:
            print(sent)
            break


def test09():
    nlp = spacy.load('en_core_web_sm')

    with open('./TextFiles/owlcreek.txt') as f:
        doc = nlp(f.read())
    sents = [sent for sent in doc.sents]
    matcher = Matcher(nlp.vocab)
    pattern = [{'LOWER': 'swimming'}, {'IS_SPACE': True, 'OP': '*'}, {'LOWER': 'vigorously'}]
    matcher.add('Swimming', None, pattern)
    found_matches = matcher(doc)

    print(doc[:36])
    print(len(doc))
    print(len(sents))
    print(sents[1].text)
    print(found_matches)
    print(doc[1265:1290])
    print(doc[3600:3615])

    for token in sents[1]:
        print(token.text, token.pos_, token.dep_, token.lemma_)
    for token in sents[1]:
        print(f'{token.text:{15}} {token.pos_:{5}} {token.dep_:{10}} {token.lemma_:{15}}')
    for sent in sents:
        if found_matches[0][1] < sent.end:
            print(sent)
            break
    for sent in sents:
        if found_matches[1][1] < sent.end:
            print(sent)
            break


def test10():
    nlp = spacy.load('en_core_web_sm')
    # doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
    # print(doc.text)
    # print(doc[4].text, doc[4].pos_, doc[4].tag_, spacy.explain(doc[4].tag_))
    # for token in doc:
    #     print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')
    #
    # doc = nlp(u'I read books on NLP.')
    # #doc = nlp(u'I read book on NLP.')
    # r = doc[1]
    # print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')

    doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
    POS_counts = doc.count_by(spacy.attrs.POS)
    print(POS_counts)
    print(doc.vocab[83].text)

    for k, v in sorted(POS_counts.items()):
        print(f'{k}. {doc.vocab[k].text:{5}}: {v}')

    TAG_counts = doc.count_by(spacy.attrs.TAG)
    for k, v in sorted(TAG_counts.items()):
        print(f'{k}. {doc.vocab[k].text:{4}}: {v}')

    DEP_counts = doc.count_by(spacy.attrs.DEP)
    for k, v in sorted(DEP_counts.items()):
        print(f'{k}. {doc.vocab[k].text:{4}}: {v}')


def test11():
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
    displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})

    for token in doc:
        print(f'{token.text:{10}} {token.pos_:{7}} {token.dep_:{7}} {spacy.explain(token.dep_)}')
    displacy.serve(doc, style='dep', options={'distance': 110})
    doc2 = nlp(u"This is a sentence. This is another, possibly longer sentence.")

    spans = list(doc2.sents)
    displacy.serve(spans, style='dep', options={'distance': 110})

    options = {'distance': 110, 'compact': 'True', 'color': 'yellow', 'bg': '#09a3d5', 'font': 'Times'}
    displacy.serve(doc, style='dep', options=options)


def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')

def remove_whitespace_entities(doc):
    doc.ents = [e for e in doc.ents if not e.text.isspace()]
    return doc


def test12():
    nlp = spacy.load('en_core_web_sm')

    doc = nlp(u'May I go to Washington, DC next May to see the Washington Monument?')
    show_ents(doc)

    doc = nlp(u'Can I please borrow 500 dollars from you to buy some Microsoft stock?')
    for ent in doc.ents:
        print(ent.text, ent.start, ent.end, ent.start_char, ent.end_char, ent.label_)

    doc = nlp(u'Tesla to build a U.K. factory for $6 million')
    show_ents(doc)
    ORG = doc.vocab.strings[u'ORG']
    new_ent = Span(doc, 0, 1, label=ORG)
    doc.ents = list(doc.ents) + [new_ent]
    show_ents(doc)

    doc = nlp(u'Our company plans to introduce a new vacuum cleaner. '
              u'If successful, the vacuum cleaner will be our first product.')
    show_ents(doc)
    matcher = PhraseMatcher(nlp.vocab)
    phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
    phrase_patterns = [nlp(text) for text in phrase_list]
    matcher.add('newproduct', None, *phrase_patterns)
    matches = matcher(doc)
    print(matches)
    PROD = doc.vocab.strings[u'PRODUCT']
    new_ents = [Span(doc, match[1], match[2], label=PROD) for match in matches]
    doc.ents = list(doc.ents) + new_ents

    doc = nlp(u'Originally priced at $29.50, the sweater was marked down to five dollars.')
    show_ents(doc)
    print(len([ent for ent in doc.ents if ent.label_ == 'MONEY']))
    doc = nlp(u'Originally priced at $29.50,\nthe sweater was marked down to five dollars.')
    show_ents(doc)
    nlp.add_pipe(remove_whitespace_entities, after='ner')

    doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")
    for chunk in doc.noun_chunks:
        print(chunk.text + ' - ' + chunk.root.text + ' - ' + chunk.root.dep_ + ' - ' + chunk.root.head.text)
    print(len(doc.noun_chunks))
    print(len(list(doc.noun_chunks)))


def test13():
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million. '
              u'By contrast, Sony sold only 7 thousand Walkman music players.')
    displacy.render(doc, style='ent', jupyter=True)
    for sent in doc.sents:
        displacy.render(nlp(sent.text), style='ent', jupyter=True)

    doc2 = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million. '
               u'By contrast, my kids sold a lot of lemonade.')
    for sent in doc2.sents:
        docx = nlp(sent.text)
        if docx.ents:
            displacy.render(docx, style='ent', jupyter=True)
        else:
            print(docx.text)
    options = {'ents': ['ORG', 'PRODUCT']}

    displacy.render(doc, style='ent', jupyter=True, options=options)
    colors = {'ORG': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)', 'PRODUCT': 'radial-gradient(yellow, green)'}

    options = {'ents': ['ORG', 'PRODUCT'], 'colors': colors}
    displacy.render(doc, style='ent', jupyter=True, options=options)
    displacy.serve(doc, style='ent', options=options)


def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i + 1].is_sent_start = True
    return doc


def split_on_newlines(doc):
    start = 0
    seen_newline = False
    for word in doc:
        if seen_newline:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text.startswith('\n'):  # handles multiple occurrences
            seen_newline = True
    yield doc[start:]  # handles the last group of tokens


def test14():
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
    doc2 = nlp(u'This is a sentence. This is a sentence. This is a sentence.')
    doc3 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')
    doc4 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')
    doc_sents = [sent for sent in doc.sents]

    for sent in doc.sents:
        print(sent)
    for token in doc2:
        print(token.is_sent_start, ' ' + token.text)
    for sent in doc3.sents:
        print(sent)
    for sent in doc4.sents:
        print(sent)
    for sent in doc3.sents:
        print(sent)

    print(doc[1])
    # print(doc.sents[1])
    print(doc_sents)
    print(doc_sents[1])
    print(type(doc_sents[1]))
    print(doc_sents[1].start, doc_sents[1].end)
    nlp.add_pipe(set_custom_boundaries, before='parser')
    print(nlp.pipe_names)
    print(doc3[7])
    doc3[7].is_sent_start = True

    nlp = spacy.load('en_core_web_sm')  # reset to the original
    mystring = u"This is a sentence. This is another.\n\nThis is a \nthird sentence."
    # SPACY DEFAULT BEHAVIOR:
    doc = nlp(mystring)
    for sent in doc.sents:
        print([token.text for token in sent])

    sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)
    nlp.add_pipe(sbd)
    doc = nlp(mystring)
    for sent in doc.sents:
        print([token.text for token in sent])


def test15():
    nlp = spacy.load('en_core_web_sm')
    with open('./TextFiles/peterrabbit.txt') as f:
        doc = nlp(f.read())

    for token in list(doc.sents)[2]:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.tag_:{6}} {spacy.explain(token.tag_)}')

    POS_counts = doc.count_by(spacy.attrs.POS)

    for k, v in sorted(POS_counts.items()):
        print(f'{k}. {doc.vocab[k].text:{5}}: {v}')

    percent = 100 * POS_counts[91] / len(doc)

    print(f'{POS_counts[91]}/{len(doc)} = {percent:{.4}}%')

    displacy.render(list(doc.sents)[2], style='dep', jupyter=True, options={'distance': 110})
    for ent in doc.ents[:2]:
        print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
    print(len([sent for sent in doc.sents]))

    list_of_sents = [nlp(sent.text) for sent in doc.sents]
    list_of_ners = [doc for doc in list_of_sents if doc.ents]

    print(len(list_of_ners))
    displacy.render(list_of_sents[0], style='ent', jupyter=True)


def test16():
    nlp = spacy.load('en_core_web_sm')
    text = u"The quick brown fox jumped over the lazy dog's back."
    doc = nlp(text)
    print(game.scorer(doc))
    print(f"{'TOKEN':{10}} {'COARSE':{8}} {'FINE':{6}} {'DESCRIPTION'}")
    print(f"{'-----':{10}} {'------':{8}} {'----':{6}} {'-----------'}")

    for token in doc:
        print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')


def main():
    test16()


if __name__ == '__main__':
    main()
