from GoogleNews import GoogleNews

# date pattern: mm/dd/yyyy
start = '04/01/2018'
end = '04/30/2018'


news = GoogleNews(lang='en', start=start, end=end)

keyword = 'frankfurt airport'
news.search(keyword)

for i in range(10):
    news.getpage(i)

print(len(news.result()))

