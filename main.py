import spacy
from spacy import displacy
import nltk
from nltk.stem.porter import *
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher


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


def main():
    test09()


if __name__ == '__main__':
    main()
