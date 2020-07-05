from GoogleNews import GoogleNews
import json


def save_to_csv(*, fields, values_list, output_file):
    with open('../../' + output_file, 'w') as output:
        seperator = ','
        new_line = '\n'
        field_list = seperator.join(fields)
        output.write(field_list + new_line)
        for values in values_list:
            output.write(seperator.join(values) + new_line)


# date pattern: mm/dd/yyyy
start = '01/01/2009'
end = '05/31/2020'
months = [[start, end], []]
pages = 2

tags = ['airplane', 'frankfurt', 'business', 'finance', 'economy', 'passengers',
        'airport', 'fraport', 'vacation', 'holiday']
fields = ['title', 'media', 'date', 'link']

month_limits = []
for year in range(2009, 2021):
    for month in range(1, 13):
        m = month
        if month < 10:
            m = '0' + str(m)
        month_limits.append(f'{m}/01/{year}')

print(month_limits)
values_list = []

for time in range(0, 2):
    start = month_limits[time]
    end = month_limits[time + 1]
    print(f'start: {start} - end: {end}')
    news = GoogleNews(lang='en', start=start, end=end)

    for tag in tags:
        news.search(tag)
        for i in range(pages):
            print(f'tag: {tag}')
            print(f'start: {start} - end: {end}')
            print(f'page: {i}')
            print('________________________________________________________________')
            news.getpage(i)
            print(len(news.result()))
            print(news.result())
            for entry in news.result():
                values = []
                values.append(entry['title'].replace(',', ' ').replace('"', ''))
                values.append(entry['media'].replace(',', ' ').replace('"', ''))
                values.append(entry['date'].replace(',', ' ').replace('"', ''))
                values.append(entry['link'].replace(',', ' ').replace('"', ''))
                values_list.append(values)

save_to_csv(fields=fields, values_list=values_list, output_file='google_news_headlines_en.csv')

