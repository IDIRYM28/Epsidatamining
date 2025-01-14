import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
import csv
def get_dataset():
    list=[]
    #recuperation des données
    file ='product.tsv'
    file_reader = pd.read_csv(file,sep='\t',encoding='utf-8')
    file_reader = file_reader[['Product Price','Product Reviews Count','Product Available Inventory']]

    #calcul des moyenne
    moyenne_prix = file_reader['Product Price'].mean()
    quantity = file_reader['Product Reviews Count'].mean()
    inventory = file_reader['Product Available Inventory'].mean()
    # remplacer les valeurs null par les moyenne
    file_reader.fillna({'Product Price': moyenne_prix, 'Product Reviews Count': quantity,'Product Available Inventory':inventory}, inplace=True)
    #normalisation des données
    dataframe = pd.DataFrame(file_reader)
    scaller = MinMaxScaler()
    data_normalized = scaller.fit_transform(dataframe)
    #application de Kmeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    file_reader ['cluster'] = kmeans.fit_predict(data_normalized)
    label_dict = {0: 'faible', 1: 'moyen', 2: 'eleve'}
    file_reader['segment'] = file_reader['cluster'].map({0: 'faible', 1: 'moyen', 2: 'élevé'})
    clusters = file_reader.groupby('cluster').mean().reset_index()
    clusters.columns = ['cluster', 'Average Price', 'Average Reviews', 'Average Inventory']
    clusters['Segment Label'] = ['Faible', 'Moyen', 'Élevé']
    print('group',clusters)
    #visualisation des clusters
    plt.scatter(file_reader['Product Price'], file_reader['Product Reviews Count'], c=file_reader['cluster'], cmap='rainbow')
    plt.xlabel('Product Price')
    plt.ylabel('Product Reviews Count')
    plt.title('Diagramme de dispersion des clusters')
    plt.show()
    fig = px.scatter(file_reader, x='Product Price', y='Product Reviews Count', color='cluster')
    fig.show()


    return list
list = get_dataset()
print (list)