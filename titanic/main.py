from NB import NAIVE_BAYES
import pandas as pd
import numpy as np

#训练
train_data = pd.read_csv("train.csv")
train_data.head()
train_label = np.array(train_data["Survived"]).tolist()
train_data = np.array(train_data[['Sex', 'Age', 'Survived']]).tolist()


#测试
test_set = pd.read_csv("test.csv")
test_data = np.array(test_set[['Sex', 'Age']]).tolist()
NB = NAIVE_BAYES(train_data, train_label)
result = []

for i in test_data:
    result.append(NB.predict(i)[0])

result = pd.DataFrame(result, columns=['Survived'])
output = pd.concat([test_set['PassengerId'], result], axis=1)

output.to_csv('prediction.csv', index=False)
print(result['Survived'].value_counts())
