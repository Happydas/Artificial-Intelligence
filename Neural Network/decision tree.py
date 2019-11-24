
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
import graphviz

csvfile = "../Data_ML/data3.csv"


print('Reading "' + csvfile + '":')
dat = np.loadtxt(csvfile, delimiter=';')
#print(dat)

df = pd.DataFrame(dat)
print(df.describe())

# Draw graph
fig, ax = plt.subplots()

for i, d in enumerate(dat[:,:2]):
    if dat[i,2] < 0:
       plt.plot(d[0], d[1], 'bo')
    else:
        plt.plot(d[0], d[1],color='#FF8000', marker='o')

plt.title('Distribution (two classes)')

plt.ylabel('Class 1 = Blue, Class -1 = Orange')
plt.xlabel('arbitrary values')
plt.show()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(dat[:,:2], dat[:,2:])

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=['x','y'],
                         class_names=['1','-1'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("2D")