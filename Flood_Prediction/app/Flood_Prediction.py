#import required libraries

import numpy as np      # for dealing high dimensional data
import pandas as pd     # to do statistical data analysis
import matplotlib.pyplot as plt   # for 2D visualization
import seaborn as sns   # High end data visualization

#read the dataset
dataset = pd.read_csv('C:/Users/91837/Desktop/Flood_Prediction/data/flood_dataset_raw.csv')

sns.displot(dataset['Temp'], kde=True)
plt.title("Temperature Distribution")
plt.show()

sns.boxplot(x=dataset['Temp'])
plt.show()

import seaborn as sns
fig = plt.gcf()
fig.set_size_inches(15, 15)

fig = sns.heatmap(dataset.corr(), annot=True, cmap='summer',
                  linewidths=1, linecolor='k', square=True,
                  mask=False, vmin=-1, vmax=1,
                  cbar_kws={"orientation": "vertical"}, cbar=True)
plt.show()

print(dataset.head())
print(dataset.info())
print(dataset.describe())






