import pandas as pd
from apyori import apriori

data = pd.read_csv(r"C:\Users\maria\Downloads\weather_nominal.csv")

transaction=[]
for i in range(len(data)):
    row_items=[]
    for j in range(len(data.columns)):
        row_items.append(str(data.values[i,j]))
    transaction.append(row_items)

rule=apriori(transaction,min_support=0.1,min_confidence=0.7,min_lift=1.0,min_length=2)
result=list(rule)

print(transaction)
print("---------------------------")
print(result)