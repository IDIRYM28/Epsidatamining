from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
sse = []
file ='product.tsv'
file_reader = pd.read_csv(file, sep='\t', encoding='utf-8')
file_reader = file_reader[['Product Price', 'Product Reviews Count', 'Product Available Inventory']]
moyenne_prix = file_reader['Product Price'].mean()
quantity = file_reader['Product Reviews Count'].mean()
inventory = file_reader['Product Available Inventory'].mean()
file_reader.fillna(
    {'Product Price': moyenne_prix, 'Product Reviews Count': quantity, 'Product Available Inventory': inventory},
    inplace=True)
dataframe = pd.DataFrame(file_reader)
scaller = MinMaxScaler()
data_normalized = scaller.fit_transform(dataframe)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_normalized)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Nombre de clusters")
plt.ylabel("SSE")
plt.show()
from sklearn.metrics import silhouette_score
silhouette_avg = []
for k in range(2, 11):
    kmeansss = KMeans(n_clusters=k)
    cluster_labels = kmeansss.fit_predict(data_normalized)
    silhouette_avg.append(silhouette_score(data_normalized, cluster_labels))
    print(silhouette_score(data_normalized, cluster_labels))
