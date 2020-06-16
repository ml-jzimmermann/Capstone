from GoogleNews import GoogleNews

# date pattern: mm/dd/yyyy
start = '04/01/2010'
end = '04/30/2010'


def save_to_csv(*, fields, values_list, output_file):
    with open('../../' + output_file, 'w') as output:
        seperator = ','
        new_line = '\n'
        field_list = seperator.join(fields)
        output.write(field_list + new_line)
        for values in values_list:
            output.write(seperator.join(values) + new_line)


news = GoogleNews(lang='en', start=start, end=end)

keyword = 'frankfurt airport'
news.search(keyword)

for i in range(10):
    news.getpage(i)

print(news.result())
