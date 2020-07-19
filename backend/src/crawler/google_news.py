from GoogleNews import GoogleNews


def save_to_csv(*, fields, values_list, output_file):
    with open(output_file, 'w') as output:
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
pages = 1

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
for time in range(0, len(month_limits) - 1):
    hashset = set()
    start = month_limits[time]
    end = month_limits[time + 1]
    print(f'start: {start} - end: {end}')
    news = GoogleNews(lang='en', start=start, end=end)

    for tag in tags:
        news.search(tag)
        for i in range(pages):
            news.getpage(i)

    for entry in news.result():
        values = []
        current_hash = hash(entry['link'])
        if not current_hash in hashset:
            values.append(entry['title'].replace(',', ' ').replace('"', '').replace('„', '').replace('“', ''))
            values.append(entry['media'].replace(',', ' ').replace('"', ''))
            values.append(entry['date'].replace(',', ' ').replace('"', ''))
            values.append(entry['link'].replace(',', ' ').replace('"', ''))
            values_list.append(values)
            hashset.add(current_hash)

save_to_csv(fields=fields, values_list=values_list, output_file='google_news_headlines_en.csv')
