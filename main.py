import wikipedia
import spacy


class TextExtractor:

    __pageTitle: str
    __pageId: str

    def __init__(self, pageTitle, pageId):
        self.__pageTitle = pageTitle
        self.__pageId = pageId

    def extract(self):
        page = wikipedia.page(title=self.__pageTitle, pageid=self.__pageId)
        f = open("./text/" + self.__pageTitle + ".txt", "w", encoding='utf-8')
        f.write(page.content)
        f.close()

    def getText(self):
        f = open("./text/" + self.__pageTitle + ".txt", "r", encoding='utf-8')
        return f.read()


class TextExtractorPipe:

    __textExtractors: [TextExtractor]

    def __init__(self):
        self.__textExtractors = []

    def addTextExtractor(self, textExtractor: TextExtractor):
        self.__textExtractors.append(textExtractor)

    def extract(self) -> str:
        result = ''
        for textExtractor in self.__textExtractors:
            result = result + textExtractor.getText()
        return result


def main():
    textExtractor1 = TextExtractor("WWII", "Q362")
    textExtractor1.extract()
    textExtractorPipe = TextExtractorPipe()
    textExtractorPipe.addTextExtractor(textExtractor1)
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), before="parser")
    doc = nlp(textExtractorPipe.extract())


if __name__ == '__main__':
    main()