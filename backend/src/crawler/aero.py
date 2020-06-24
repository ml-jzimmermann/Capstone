import requests as http
from bs4 import BeautifulSoup as soup


def get_aero_news(past_pages=0):
    url = 'https://www.adv.aero/pressemeldungen/'
    parsed_articles = _parse_aero_news_url(url)

    for i in range(past_pages):
        url = url + 'page/' + str(i) + '/'
        parsed_articles.extend(_parse_aero_news_url(url))
    return parsed_articles


def _parse_aero_news_url(url):
    parsed_articles = []
    html = http.get(url).text
    doc = soup(html, 'html.parser')
    articles = doc.find_all('article', {'class': 'post'})
    for article in articles:
        anker = article.find_all('a')[0]
        href = anker['href']
        parsed_articles.append(_parse_aero_news_article(href))
    return parsed_articles


def _parse_aero_news_article(article_url):
    html = http.get(article_url).text
    doc = soup(html, 'html.parser')
    article = doc.find_all('article', {'class': 'post'})[0]
    headline = article.find_all('h1', {'class': 'entry-title'})[0].text.strip()
    date = article.find_all('div', {'class': 'entry-date'})[0].text.strip()
    content = article.find_all('div', {'class': 'entry-content'})[0].text.strip()
    cats = article.find_all('div', {'class': 'entry-cats'})[0].text.strip()
    author = article.find_all('div', {'class': 'entry-author'})[0].text.strip()
    return (headline, content, date, cats, author)


get_aero_news(past_pages=10)
