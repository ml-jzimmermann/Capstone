import csv

class CSVwriter:


    csvFile=None
    writer=None
    def __init__(self):
        self.csvFile = None
        self.writer = None

    def openFile(self, filename):
        try:
            self.csvFile = open(filename, 'w+')
            self.writer = csv.writer(self.csvFile)
            self.writer.writerow(('Link', 'Text', 'Headline', 'Datum', 'Zeitschrift', 'Rubrik', 'Land', 'Land(Herkunft)', 'Stadt'))
        except Exception as e:
            print('OpenError')
            print(e)

    def writeRow(self,Link,Text, headLine, Datum, Zeitschrift, Rubrik, Land, LandH, Stadt):
        try:
              self.writer.writerow((Link, Text, headLine, Datum, Zeitschrift, Zeitschrift, Rubrik, Land, LandH, Stadt))
        except Exception as e:
            print('WritingError')
            print(e)


    def closeFile(self):
        try:
            self.csvFile.close()
        except Exception as e:
            print('ClosingError')
            print(e)