from parseArticle import *
from csvWriter import CSVwriter

link = 'https://www.airliners.de/ryanair-sucht-alternativen-zum-bbi/20477'
parsedHTML = getArticleSite(link)
headLine = getArticleHeadline(parsedHTML)
date = getDate(parsedHTML)
text = getArticleText(parsedHTML)
write = CSVwriter()
write.openFile('test.csv')
write.writeRow(link, text, headLine,date, 'Airliner.de', 'Allgemein', 'Global', 'Deutschland', 'Berlin')
write.closeFile()