import wikipedia


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


def main():
    textExtractor1 = TextExtractor("WWII", "Q362")
    textExtractor1.extract()


if __name__ == '__main__':
    main()