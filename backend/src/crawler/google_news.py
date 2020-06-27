from GoogleNews import GoogleNews


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
pages = 5

tags = ['airplane', 'frankfurt', 'business', 'finance', 'economy', 'passengers',
        'airport', 'fraport', 'vacation', 'holiday']

news = GoogleNews(lang='de', start=start, end=end)
keyword = 'frankfurt airport'
for tag in tags:
    news.search(tag)


for i in range(pages):
    news.getpage(i)

print(news.result())
