import requests as http
from bs4 import BeautifulSoup as soup


def get_telegraph_news(past_pages=0):
    categories = ['aircraft', 'airlines', 'airports', 'pax-ex', 'safety', 'sustainability']
    for category in categories:
        url = 'https://www.aerotelegraph.com/en/category/' + str(category)
        parsed_news = _parse_telegraph_news_category(url)

def _parse_telegraph_news_category(category_url):
    pass
