import pandas as pd

data = pd.read_csv('part1.csv')
pd.set_option('display.max_columns', 9)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)

#leere Zeilen prüfen
print ('Leere Zeilen: ', data.isnull().sum())
print(data.shape)

matches = ['Stellenanzeige','stellenanzeige']
count = 0
index = []
for i in range(len(data)):
    for y in [0,1]:
        if any(str (data.iloc[i,y]).find(x)!= -1 for x in matches):
            index.append(i)
            count += 1
        break

print('index: ', len(index))

data_clean = data.drop(index=index)
print(data_clean.shape)
print('total count2: ', count)

data_clean.to_csv('data_clean.csv')





