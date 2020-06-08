from parseArticle import *
from csvWriter import CSVwriter
import re
import time

def crawl(endpos):
    write = CSVwriter()
    write.openFile('airliners.csv')
    for i in range(1,endpos):
        print("Search for articles on:")
        print('\thttps://www.airliners.de/ticker?page='+str(i))
        html = getArticleSite('https://www.airliners.de/ticker?page='+str(i))
        for ticker in html.find_all('div',{'class':'ticker'}):
            #TODO evtl. entfernen je nach performance in anderem netzwerk
            time.sleep(1)
            for link in ticker.find_all('a', href = re.compile('https://www.airliners.de/*')):
                if 'href' in link.attrs:
                    # print(link.attrs['href'])
                    # print("get Site")
                    parsedHTML = getArticleSite(link.attrs['href'])
                    # print("get Headline")
                    headLine = getArticleHeadline(parsedHTML)
                    # print("get date")
                    date = getDate(parsedHTML)
                    # print("get text")
                    text = getArticleText(parsedHTML)
                    # print("write to file")
                    if 'Exklusiv für airliners+ Abonnenten' not  in text:
                        print("\t\tadd article from link: "+link.attrs['href'])
                        write.writeRow(link.attrs['href'], text, headLine,date, 'Airliner.de', 'Allgemein', 'Global', 'Deutschland', 'Berlin')



    write.closeFile()

#endpos immer minus 1
crawl(2)
# crawl(228)