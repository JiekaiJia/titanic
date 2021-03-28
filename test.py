
test = [
    ['1',2,3],
    [2,1,6],
    [6,4,3],
    [7,9,6],
    [3,3,8],
    [2,8,7],
    [8,6,1],
    [9,7,2],
]

fileObject = open('sampleList.txt', 'w')
for ip in test:
    ii = ip[0:2]
    fileObject.write(str(ii))
    fileObject.write('\n')
fileObject.close()

#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")
#print(sorted(test, key=lambda x:x[2]))