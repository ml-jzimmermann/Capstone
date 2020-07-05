# clean files
with open('airliner_completed.csv', 'w') as output:
    with open('part1.csv', 'rb') as f:
        lines = f.readlines()
        for l in lines:
            line = l.decode('utf-8', 'ignore')
            output.write(line)

    with open('part2.csv', 'rb') as f:
        lines = f.readlines()
        for l in lines:
            if l.decode('utf-8', 'ignore').startswith('Link,Text,Headline,'):
                continue
            line = l.decode('utf-8', 'ignore')
            output.write(line)

    with open('part3.csv', 'rb') as f:
        lines = f.readlines()
        for l in lines:
            if l.decode('utf-8', 'ignore').startswith('Link,Text,Headline,'):
                continue
            line = l.decode('utf-8', 'ignore')
            output.write(line)
