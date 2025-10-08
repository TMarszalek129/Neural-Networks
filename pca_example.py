# Example based on https://github.com/MamczurMiroslaw/PCA

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

seed = 2019

def apply_scalers(df, columns_to_exclude=None):
    if columns_to_exclude:
        exclude_filter = ~df.columns.isin(columns_to_exclude)
    else:
        exclude_filter = ~df.columns.isin([])
    for column in df.iloc[:, exclude_filter].columns:
        df[column] = df[column].astype(float)

    df.loc[:, exclude_filter] = StandardScaler().fit_transform(df.loc[:, exclude_filter])
    return df

df = pd.read_excel('datasets\data.xlsx', sheet_name='dataframe')
df = apply_scalers(df, columns_to_exclude=['Nazwa'])
df.sort_values(by='2019Q2_aktywa', ascending=False).head()

columns = ~df.columns.isin(['Nazwa'])
pca = PCA(n_components=3)

principal_components = pca.fit_transform(df.loc[:, columns])
principal_df = pd.DataFrame(data=principal_components,
                            columns=['principal component 1', 'principal component 2', 'principal component 3'])
principal_df.insert(loc=0, column="Nazwa", value=df['Nazwa'])

trace0 = go.Scatter(
    x=principal_df['principal component 1'],
    y=principal_df['principal component 3'],
    text=principal_df['Nazwa'],
    textposition="top center",
    name='Piony',
    mode='markers+text',
    marker=dict(
        size=10,
        color='rgb(228,26,28)',
        line=dict(
            width=1,
            color='rgb(0, 0, 0)'
        )
    )
)

data = [trace0]

layout = dict(title='Podobieństwo Banków na podstawie PCA',
              yaxis=dict(zeroline=False, title='PC2 (principal component 2)'),
              xaxis=dict(zeroline=False, title='PC1 (principal component 1)')
              )

fig = dict(data=data, layout=layout)
iplot(fig, filename='styled-scatter')
