import pandas as pd
from datetime import datetime as dt


df = pd.read_csv('../../data/passagierzahlen.csv').values
for line in df:
    date = dt.strptime(line[0], '%d/%m/%Y')
    print(date)
