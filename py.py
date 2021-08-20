import streamlit as st
st.set_page_config(page_title='Code Healthscape App Engine', initial_sidebar_state = 'auto')

st.image(
    "hero.png",
    width=800,
)

import streamlit.components.v1 as components

st.image("https://img.shields.io/badge/build-passing-brightgreen")



import pandas as pd
import ipywidgets as wg
from IPython.display import display, clear_output, Image
from ipywidgets import *
import plotly.express as px
import re

st.sidebar.title('Navigation')
nav = st.sidebar.radio("Go To", ('Database Library', 'Predictive', 'Search by project', 'Search Reference', 'Building codes', 'Add a project'), index=0, key=None, help=None, on_change=None, args=None, kwargs=None)

st.sidebar.title('Logs')
df = pd.read_csv("analysis5.csv")

import sys
b = sys.getsizeof(df)

import math
def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])
st.sidebar.markdown('Dataframe loaded.  \nProcessed ' + str(len(df)) + ' entries in table.  \nMem cache: ' + str(convert_size(b))) 

n = len(pd.unique(df['Project']))
st.sidebar.markdown('Number of projects in dataframe: ' + str(n))

st.sidebar.title('Contribute')
# st.sidebar.title('About')
# st.sidebar.info('This app is maintained by Ansh Sharma at the Healthcare group at FivD')
st.sidebar.image("fivdw.png")
st.sidebar.info('This an open source project and you are very welcome to **contribute** your awesome comments, questions, resources and apps as issues of or pull requests to the source code.')

if nav == 'Database Library':
    st.title('Requirements')
    col1, col2, col3 = st.columns(3)
    with col1:
    	s = st.selectbox(
    	'Sort input by',
    	('Beds', 'TotalBldg_GSF'))
    with col2:
    	s3 = st.number_input("Enter your requirement for " + str(s), step=100, format=None)
    with col3:
    	s4 = st.number_input("Enter your deviation", step=1, format=None, value=100)

    df2 = df[s].value_counts()
    if s == 'Beds':
    	xT = 'Number of Beds'
    else:
        xT = 'Area (sqft)'

    df2 = df2.to_frame()
    df3 = df2.reset_index()
    figh = px.histogram(df3, x="index", nbins=20, template='none', marginal='rug', color_discrete_sequence=px.colors.sequential.RdBu,
                    labels={
                         "index": xT
                     },
                    title="Number of Healthcare projects undertaken by us till date")

    figh.update_layout({
    "plot_bgcolor": "rgba(0, 0, 0, 0)",
    "paper_bgcolor": "rgba(0, 0, 0, 0)",
    })

    st.write(figh)

    s1_opt = list(df['Cat_Major'].unique())
    s1 = st.selectbox('Tab: ', options=s1_opt)

    df_temp = df.loc[df['Cat_Major'] == s1]

    s2 = st.multiselect(
        'Deparment Requirements: ',
        df_temp['Cat_Minor'].unique())

    def convert(s2):
        return tuple(s2)
    # st.write(convert(s2))

    df_query = df.loc[(df[s] >= int(s3)-int(s4)) & (df[s] < int(s3)+int(s4))]
    df4 = df_query.loc[(df['Cat_Minor'].isin(convert(s2))) | (df['Cat_Minor'] == convert(s2))]
    # df4 = df.loc[df['Cat_Minor'].isin(convert(s2)) & df['Project'].isin(s3)]
    # If you also want to filter out projects, this shows mean now
    df5 = df4[['Cat_Minor', 'Total_SF']]
    df6 = df5.groupby(['Cat_Minor'], as_index=False).mean()
    df7 = df6.style.set_properties(**{'text-align': 'left'})
    figp = px.pie(df6, values='Total_SF', names='Cat_Minor', title='Area Allocation (Mean)', hole=.5, color_discrete_sequence=px.colors.sequential.RdBu)

    st.title('Spatial breakdown')
    st.write(figp)
    st.table(df7)

    st.subheader('Queried dataframe:')
    df8 = df4[['Project', 'Beds', 'TotalBldg_GSF']]
    df9 = df8.groupby(['Project'], as_index=False).mean()
    df10 = df9.style.set_properties(**{'text-align': 'left'})
    st.table(df10)

elif nav == 'Predictive':
    import matplotlib.pyplot as plt
    import numpy as np
    data = pd.read_csv('test.csv', index_col=False, header=0)
    X = data.c1.values
    y = data.c2.values
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fitting Linear Regression to the dataset
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    # Fitting Polynomial Regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=3)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)

    st.title('ML Framework')
    ml = st.selectbox('Choose Model: ', options=['Linear Regression', 'Polynomial Regression', 'KNN', 'Decision Tree', 'Random Forests', 'Neutral Network'])
    if ml == 'Linear Regression':
        # Visualizing the Linear Regression results
        def viz_linear():
            plt.scatter(X, y, color='red')
            plt.plot(X, lin_reg.predict(X), color='blue')
            plt.title('Truth or Bluff (Linear Regression)')
        #     plt.xlabel('Position level')
        #     plt.ylabel('Salary')
            plt.xticks(())
            plt.yticks(())
            # plt.show()
            return
        viz_linear()

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plt)
        with col2:
            mlin = st.number_input("Enter your requirement for Beds: ", step=1, format=None)
            # Predicting a new result with Linear Regression
            st.latex(str(lin_reg.predict([[mlin]])).lstrip('[').rstrip(']') + "  sqft.")
            st.write('Closest matching project: ')
            df_match = df.loc[df['Beds'] == mlin]
            df_matcho = df_match.drop_duplicates(subset = ["Project"])
            st.write(df_matcho[['Project', 'Beds', 'TotalBldg_GSF']])
        plt.show
    elif ml == 'Polynomial Regression':
        # Visualizing the Polymonial Regression results
        def viz_polymonial():
            plt.scatter(X, y, color='red')
            plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
            plt.title('Truth or Bluff (Polynomial Regression)')
        #     plt.xlabel('Position level')
        #     plt.ylabel('Salary')
            plt.xticks(())
            plt.yticks(())
            # plt.show()
            return
        viz_polymonial()
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plt)
        with col2:
            mlin = st.number_input("Enter your requirement for Beds: ", step=1, format=None)
            # Predicting a new result with Polynomial Regression
            st.latex(str(pol_reg.predict(poly_reg.fit_transform([[mlin]]))).lstrip('[').rstrip(']') + "  sqft.")
            st.write('Closest matching project: ')
            df_match = df.loc[df['Beds'] == mlin]
            df_matcho = df_match.drop_duplicates(subset = ["Project"])
            st.write(df_matcho[['Project', 'Beds', 'TotalBldg_GSF']])
        plt.show
    else:
        st.write('Work in progress.')
elif nav == 'Search by project':
    import re

    s1_opt = list(df['Cat_Major'].unique())
    s1 = st.selectbox('Tab: ', options=s1_opt)

    df_temp = df.loc[df['Cat_Major'] == s1]

    s2 = st.multiselect(
        'Deparment Requirements: ',
        df_temp['Cat_Minor'].unique())

    def convert(s2):
        return tuple(s2)
    df_specific = df.loc[(df['Cat_Minor'].isin(convert(s2))) | (df['Cat_Minor'] == convert(s2))]
    df_specific2 = df_specific[['Cat_Minor', 'Total_SF', 'Project']]
    df_specific3 = df_specific2.groupby(['Project', 'Cat_Minor'], as_index=False).mean()
    x = st.selectbox(
        'Search for project: ',
        df['Project'].unique())
    def convert(x):
        return tuple(x)

    df_specific3.dropna(axis=1, thresh=1)
    df_final = df_specific3.loc[df_specific3['Project'].isin([x])]

    # df_final = df_specific3['Project'].isin([x])
    # df_final = df_specific3['Project'].str.contains(x)
    st.table(df_final)
    fig = px.pie(df_final, values='Total_SF', names='Cat_Minor', title='Area Allocation: '+str(df_final['Project'].unique()).lstrip('[').rstrip(']'), hole=.5, color_discrete_sequence=px.colors.sequential.Agsunset)
    st.write(fig)

    # fig = px.pie(df_specific3.loc[(df_specific3['Project'].isin(convert(x)))], values='Total_SF', names='Cat_Minor', title='Area Allocation: '+str(df_specific3.loc[(df_specific3['Project'].isin(convert(x)))]['Project'].unique()), hole=.5, color_discrete_sequence=px.colors.sequential.Agsunset)
    # st.write(fig)
    # dfs = df[['Project', 'TotalBldg_GSF', 'Beds']]
    # st.write(dfs.loc[(df_specific3['Project'].isin(convert(x)))].groupby(['Project'], as_index=False).mean())

elif nav == 'Search Reference':
    import os
    from IPython.display import Image
    xx = st.selectbox(
        'Open Reference :',
        tuple(os.listdir('img/')))
    st.image('img/'+xx)

elif nav == 'Building codes':
    yo = """
    <svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 554.73 653.04"><defs><style>.cls-1{fill:#fff;}.cls-1,.cls-2{stroke:#000;}.cls-1,.cls-2,.cls-3,.cls-4,.cls-5{stroke-miterlimit:10;}.cls-2,.cls-3,.cls-4,.cls-5{fill:none;}.cls-3,.cls-4,.cls-5{stroke:#c1272d;}.cls-4{stroke-dasharray:11.46 11.46;}.cls-5{stroke-dasharray:12.04 12.04;}</style></defs><rect class="cls-1" x="39.21" y="3.63" width="120.72" height="1.96"/><rect class="cls-1" x="37.64" y="0.5" width="1.57" height="5.09"/><rect class="cls-1" x="159.92" y="0.5" width="1.57" height="5.09"/><rect class="cls-2" x="60.14" y="38.72" width="58.96" height="145.83"/><rect class="cls-2" x="119.1" y="52.54" width="2.35" height="49.04"/><rect class="cls-2" x="57.79" y="52.41" width="2.35" height="49.04"/><rect class="cls-2" x="119.1" y="121.15" width="2.35" height="49.04"/><rect class="cls-2" x="57.79" y="121.02" width="2.35" height="49.04"/><polyline class="cls-2" points="51.6 6.04 51.6 15.44 65.29 15.44 67.64 13.15 67.64 6.04"/><rect class="cls-2" x="64.84" y="42.63" width="48.78" height="25.43" rx="5.2"/><line class="cls-2" x1="60.14" y1="104.98" x2="119.1" y2="104.98"/><line class="cls-2" x1="60.14" y1="124.74" x2="119.1" y2="124.74"/><polyline class="cls-3" points="184.41 211.19 184.41 217.19 178.41 217.19"/><line class="cls-4" x1="166.95" y1="217.18" x2="12.23" y2="217.18"/><polyline class="cls-3" points="6.5 217.19 0.5 217.19 0.5 211.19"/><line class="cls-5" x1="0.5" y1="199.14" x2="0.5" y2="12.52"/><polyline class="cls-3" points="0.5 6.5 0.5 0.5 6.5 0.5"/><line class="cls-4" x1="17.96" y1="0.5" x2="172.68" y2="0.5"/><polyline class="cls-3" points="178.41 0.5 184.41 0.5 184.41 6.5"/><line class="cls-5" x1="184.41" y1="18.54" x2="184.41" y2="205.16"/><path d="M144.48,306.57h3.74v.8h-2.67v1.71h2.31v.78h-2.31v2.31h-1.07Z" transform="translate(-75.22 -109.26)"/><path d="M152.4,310.2a2,2,0,1,1-4,0v-.06a2,2,0,1,1,4,0Zm-1,0v-.09c0-1-.39-1.43-1-1.43s-1,.46-1,1.43v.09c0,1,.4,1.43,1,1.43S151.38,311.18,151.38,310.22Z" transform="translate(-75.22 -109.26)"/><path d="M157,310.2a2,2,0,1,1-4,0v-.06a2,2,0,1,1,4,0Zm-1,0v-.09c0-1-.39-1.43-1-1.43s-1,.46-1,1.43v.09c0,1,.39,1.43,1,1.43S156,311.18,156,310.22Z" transform="translate(-75.22 -109.26)"/><path d="M157.87,311v-2.15h-.6v-.65h.6V307h1v1.15H160v.65h-1.15V311c0,.38.13.52.47.52a5.6,5.6,0,0,0,.65-.06v.69a4.43,4.43,0,0,1-.87.09C158.2,312.22,157.87,311.86,157.87,311Z" transform="translate(-75.22 -109.26)"/><path d="M166.25,310.2a2,2,0,1,1-4,0v-.06a2,2,0,1,1,4,0Zm-1,0v-.09c0-1-.39-1.43-1-1.43s-1,.46-1,1.43v.09c0,1,.39,1.43,1,1.43S165.23,311.18,165.23,310.22Z" transform="translate(-75.22 -109.26)"/><path d="M168.19,308.83v3.34h-1v-3.34h-.6v-.65h.6v-.35c0-.87.39-1.32,1.38-1.32a3,3,0,0,1,.59.06v.65a2.08,2.08,0,0,0-.42,0c-.4,0-.56.15-.56.53v.46h1v.65Z" transform="translate(-75.22 -109.26)"/><path d="M172.44,311.45v.72h-1v-5.6h1v2.32a1.42,1.42,0,0,1,1.31-.83c1,0,1.67.9,1.67,2.07v.09a1.85,1.85,0,0,1-1.67,2.07A1.43,1.43,0,0,1,172.44,311.45Zm2-1.32c0-1-.39-1.43-1-1.43s-1,.54-1,1.43v.09c0,.87.44,1.43,1,1.43s1-.47,1-1.43Z" transform="translate(-75.22 -109.26)"/><path d="M176,310.2v-.06a1.93,1.93,0,0,1,2-2.08c1.22,0,1.89.76,1.89,2.09v.18H177c0,.88.43,1.32,1.07,1.32.35,0,.69-.21,1-.78l.73.32a2,2,0,0,1-3.82-1Zm1-.43h1.82c-.06-.77-.34-1.08-.88-1.08S177.07,309.06,177,309.77Z" transform="translate(-75.22 -109.26)"/><path d="M183.34,311.45a1.42,1.42,0,0,1-1.31.84c-1,0-1.67-.91-1.67-2.07v-.09c0-1.17.67-2.07,1.67-2.07a1.41,1.41,0,0,1,1.31.83v-2.32h1v5.6h-1Zm-2-1.32v.09c0,1,.39,1.43,1,1.43s1-.56,1-1.43v-.09c0-.89-.44-1.43-1-1.43S181.38,309.16,181.38,310.13Z" transform="translate(-75.22 -109.26)"/><path d="M212.57,269.49v4.76h-1.06v-4.76h-1.66v-.84h4.39v.84Z" transform="translate(-75.22 -109.26)"/><path d="M214.41,274.25v-4h1v.95a1.32,1.32,0,0,1,1.29-1h.17v.93l-.33,0c-.7,0-1.13.39-1.13,1.25v1.88Z" transform="translate(-75.22 -109.26)"/><path d="M219.68,273.6a1.37,1.37,0,0,1-1.31.75,1.22,1.22,0,0,1-1.35-1.19c0-.73.48-1.31,2.3-1.37l.33,0v-.14c0-.55-.21-.8-.72-.8a.82.82,0,0,0-.78.62l-.88-.16a1.62,1.62,0,0,1,1.66-1.15,1.51,1.51,0,0,1,1.71,1.73v2.38h-1Zm0-.87v-.42h-.26c-1.11.07-1.38.39-1.38.8a.6.6,0,0,0,.67.59A.93.93,0,0,0,219.65,272.73Z" transform="translate(-75.22 -109.26)"/><path d="M221.46,274.25v-4h1V271a1.38,1.38,0,0,1,1.32-.89,1.26,1.26,0,0,1,1.33,1.42v2.69h-1v-2.5c0-.58-.21-.86-.7-.86s-1,.46-1,1.19v2.17Z" transform="translate(-75.22 -109.26)"/><path d="M225.63,273.28l.66-.4a1.32,1.32,0,0,0,1.21.85c.52,0,.82-.22.82-.51s-.16-.4-.71-.52l-.55-.11c-.83-.17-1.21-.6-1.21-1.21s.63-1.24,1.63-1.24a1.9,1.9,0,0,1,1.73.93l-.63.39a1.23,1.23,0,0,0-1.06-.68c-.46,0-.72.19-.72.47s.13.41.59.51l.55.11c1,.2,1.35.61,1.35,1.25s-.71,1.25-1.77,1.25A1.94,1.94,0,0,1,225.63,273.28Z" transform="translate(-75.22 -109.26)"/><path d="M231.2,270.92v3.33h-1v-3.33h-.6v-.66h.6v-.35c0-.86.39-1.32,1.38-1.32a3,3,0,0,1,.59.06v.65l-.42,0c-.4,0-.56.15-.56.53v.46h1v.66Z" transform="translate(-75.22 -109.26)"/><path d="M232.44,272.28v0a1.93,1.93,0,0,1,2-2.09c1.22,0,1.88.76,1.88,2.1v.17h-2.84c0,.89.42,1.32,1.06,1.32.36,0,.7-.21,1-.77l.73.32a1.91,1.91,0,0,1-1.77,1.09A1.93,1.93,0,0,1,232.44,272.28Zm1-.43h1.82c-.05-.77-.33-1.08-.88-1.08S233.55,271.15,233.46,271.85Z" transform="translate(-75.22 -109.26)"/><path d="M237,274.25v-4h1v.95a1.32,1.32,0,0,1,1.29-1h.17v.93l-.32,0c-.71,0-1.14.39-1.14,1.25v1.88Z" transform="translate(-75.22 -109.26)"/><line class="cls-2" x1="89.62" y1="189.04" x2="89.62" y2="196.45"/><polygon points="87.77 190.44 89.62 189.65 91.47 190.44 89.62 186.05 87.77 190.44"/><line class="cls-2" x1="89.62" y1="212.33" x2="89.62" y2="204.92"/><polygon points="91.47 210.93 89.62 211.72 87.77 210.93 89.62 215.32 91.47 210.93"/><path d="M144.48,126.1h1.07v2.33h2.32V126.1h1.07v5.6h-1.07v-2.4h-2.32v2.4h-1.07Z" transform="translate(-75.22 -109.26)"/><path d="M149.72,129.74v-.06a1.93,1.93,0,0,1,2-2.09c1.22,0,1.89.76,1.89,2.1v.17h-2.85c0,.89.43,1.32,1.07,1.32.36,0,.69-.21,1-.77l.73.32a2,2,0,0,1-3.82-1Zm1-.44h1.82c-.06-.76-.34-1.08-.88-1.08S150.83,128.6,150.75,129.3Z" transform="translate(-75.22 -109.26)"/><path d="M156.66,131.05a1.37,1.37,0,0,1-1.31.76,1.23,1.23,0,0,1-1.35-1.2c0-.73.48-1.31,2.31-1.37l.32,0v-.14c0-.54-.21-.8-.72-.8a.82.82,0,0,0-.78.62l-.88-.16a1.62,1.62,0,0,1,1.66-1.15,1.51,1.51,0,0,1,1.71,1.73v2.38h-1Zm0-.87v-.42l-.26,0c-1.11.06-1.38.38-1.38.79a.61.61,0,0,0,.68.6A.93.93,0,0,0,156.63,130.18Z" transform="translate(-75.22 -109.26)"/><path d="M161.26,131a1.42,1.42,0,0,1-1.31.84c-1,0-1.67-.91-1.67-2.07v-.09c0-1.16.67-2.07,1.67-2.07a1.41,1.41,0,0,1,1.31.83V126.1h1v5.6h-1Zm-2-1.32v.09c0,1,.39,1.43,1,1.43s1-.56,1-1.43v-.09c0-.88-.44-1.43-1-1.43S159.3,128.7,159.3,129.66Z" transform="translate(-75.22 -109.26)"/><path d="M164.53,130.73l.66-.4a1.33,1.33,0,0,0,1.21.85c.52,0,.82-.22.82-.51s-.16-.4-.71-.52L166,130c-.83-.17-1.21-.6-1.21-1.21s.63-1.24,1.64-1.24a1.89,1.89,0,0,1,1.72.93l-.63.39a1.23,1.23,0,0,0-1.06-.68c-.46,0-.72.19-.72.47s.13.41.59.51l.55.11c1,.2,1.35.62,1.35,1.25s-.71,1.25-1.77,1.25A1.94,1.94,0,0,1,164.53,130.73Z" transform="translate(-75.22 -109.26)"/><path d="M168.92,127.14v-.94h1v.94Zm0,4.56v-4h1v4Z" transform="translate(-75.22 -109.26)"/><path d="M173.63,131a1.44,1.44,0,0,1-1.32.84c-1,0-1.66-.91-1.66-2.07v-.09c0-1.16.66-2.07,1.66-2.07a1.43,1.43,0,0,1,1.32.83V126.1h1v5.6h-1Zm-2-1.32v.09c0,1,.39,1.43,1,1.43s1-.56,1-1.43v-.09c0-.88-.44-1.43-1-1.43S171.67,128.7,171.67,129.66Z" transform="translate(-75.22 -109.26)"/><path d="M175.31,129.74v-.06a1.93,1.93,0,0,1,2-2.09c1.23,0,1.89.76,1.89,2.1v.17h-2.85c0,.89.43,1.32,1.07,1.32.36,0,.69-.21,1-.77l.73.32a2,2,0,0,1-3.82-1Zm1-.44h1.82c0-.76-.33-1.08-.88-1.08S176.43,128.6,176.34,129.3Z" transform="translate(-75.22 -109.26)"/><line class="cls-2" x1="89.62" y1="8.57" x2="89.62" y2="15.98"/><polygon points="87.77 9.98 89.62 9.19 91.47 9.98 89.62 5.59 87.77 9.98"/><line class="cls-2" x1="89.62" y1="31.87" x2="89.62" y2="24.46"/><polygon points="91.47 30.46 89.62 31.25 87.77 30.46 89.62 34.85 91.47 30.46"/><line class="cls-2" x1="177.43" y1="162.25" x2="170.02" y2="162.25"/><polygon points="176.03 160.4 176.81 162.25 176.03 164.1 180.41 162.25 176.03 160.4"/><line class="cls-2" x1="124.63" y1="162.25" x2="132.04" y2="162.25"/><polygon points="126.04 164.1 125.25 162.25 126.04 160.4 121.65 162.25 126.04 164.1"/><path d="M90.61,268.65h3.74v.8H91.67v1.71H94V272H91.67v2.3H90.61Z" transform="translate(-75.22 -109.26)"/><path d="M97,273.6a1.36,1.36,0,0,1-1.3.75,1.23,1.23,0,0,1-1.36-1.19c0-.73.48-1.31,2.31-1.37l.33,0v-.14c0-.55-.22-.8-.72-.8a.82.82,0,0,0-.79.62l-.88-.16a1.62,1.62,0,0,1,1.67-1.15,1.5,1.5,0,0,1,1.7,1.73v2.38H97Zm0-.87v-.42h-.27c-1.11.07-1.37.39-1.37.8a.59.59,0,0,0,.67.59A.93.93,0,0,0,96.94,272.73Z" transform="translate(-75.22 -109.26)"/><path d="M98.74,274.25v-4h1v.95a1.32,1.32,0,0,1,1.29-1h.17v.93l-.33,0c-.7,0-1.13.39-1.13,1.25v1.88Z" transform="translate(-75.22 -109.26)"/><path d="M103.26,273.28l.65-.4a1.34,1.34,0,0,0,1.22.85c.52,0,.81-.22.81-.51s-.16-.4-.7-.52l-.55-.11c-.83-.17-1.21-.6-1.21-1.21s.62-1.24,1.63-1.24a1.9,1.9,0,0,1,1.73.93l-.63.39a1.25,1.25,0,0,0-1.07-.68c-.45,0-.72.19-.72.47s.14.41.6.51l.55.11c1,.2,1.35.61,1.35,1.25s-.71,1.25-1.78,1.25A1.92,1.92,0,0,1,103.26,273.28Z" transform="translate(-75.22 -109.26)"/><path d="M107.65,269.69v-.94h1v.94Zm0,4.56v-4h1v4Z" transform="translate(-75.22 -109.26)"/><path d="M112.35,273.53a1.42,1.42,0,0,1-1.31.84c-1,0-1.66-.91-1.66-2.07v-.09c0-1.17.66-2.07,1.66-2.07a1.41,1.41,0,0,1,1.31.83v-2.32h1v5.6h-1Zm-2-1.32v.09c0,1,.39,1.43,1,1.43s1-.56,1-1.43v-.09c0-.89-.44-1.43-1-1.43S110.39,271.24,110.39,272.21Z" transform="translate(-75.22 -109.26)"/><path d="M114,272.28v0a1.93,1.93,0,0,1,2-2.09c1.22,0,1.89.76,1.89,2.1v.17H115c0,.89.42,1.32,1.06,1.32.36,0,.7-.21,1-.77l.72.32a1.91,1.91,0,0,1-1.77,1.09A1.93,1.93,0,0,1,114,272.28Zm1-.43h1.83c-.06-.77-.34-1.08-.88-1.08S115.15,271.15,115.06,271.85Z" transform="translate(-75.22 -109.26)"/><line class="cls-2" x1="54.81" y1="162.25" x2="47.4" y2="162.25"/><polygon points="53.4 160.4 54.19 162.25 53.4 164.1 57.79 162.25 53.4 160.4"/><line class="cls-2" x1="5.01" y1="162.25" x2="12.42" y2="162.25"/><polygon points="6.42 164.1 5.63 162.25 6.42 160.4 2.03 162.25 6.42 164.1"/></svg>
    """
    st.markdown(yo, unsafe_allow_html=True)
    codes = st.selectbox(
    'Select type: ',
    ['Medical/surgical patient rooms', 'Critical care patient rooms', 'Rooms for patients of size'])
    if codes == 'Medical/surgical patient rooms':
        st.latex('Head side = , Foot of bed = 36, Transfer = 48, Far side = 36')
    elif codes == 'Critical care patient rooms':
        st.latex('Head side = 18, Foot of bed = 36, Transfer = 66, Far side = 54')
    elif codes == 'Rooms for patients of size':
        st.latex('Head side = , Foot of bed = 60, Transfer = 72, Far side = 60')

elif nav == 'Add a project':
    import streamlit as st; import pandas as pd; import numpy as np; import spacy; import re; import json; import hashlib

    st.title('Add a new project')

    uploaded_file = st.file_uploader("Upload your new project data with style", type=['csv'])

    @st.cache
    def load_data(nrows):
        df = pd.read_csv(uploaded_file, nrows=nrows)
        en_core = spacy.load('en_core_sci_md')

        #str
        df['Name'] = df['Name'].astype(str)
        df['City'] = df['Name'].astype(str)

        # Regex replace
        df['Department'] = df['Department'].str.replace('[^A-Za-z]', '')
        df['Name'] = df['Name'].str.replace('[^A-Za-z]', '')

        # Normalization
        df['Department'] = [entry.lower() for entry in df['Department']]
        df['Name'] = [entry.lower() for entry in df['Name']]
        df['City'] = [entry.lower() for entry in df['City']]
        df['Project Name'] = [entry.lower() for entry in df['Project Name']]

        # Lemmatization
        df['Name'] = df['Name'].apply(lambda x: " ".join([y.lemma_ for y in en_core(x)]))

        # # Append Location Data
        # longitude = []
        # latitude = []
        # def findGeocode(city):
        #   try:
        #       geolocator = Nominatim(user_agent="your_app_name")
        #       return geolocator.geocode(city)
        #   except GeocoderTimedOut:
        #       return findGeocode(city)
        # for i in (df["City"]):
        #   if findGeocode(i) != None:
        #       loc = findGeocode(i)
        #       latitude.append(loc.latitude)
        #       longitude.append(loc.longitude)
        #   else:
        #       latitude.append(np.nan)
        #       longitude.append(np.nan)
        # df["longitude"] = longitude
        # df["latitude"] = latitude
        return df



    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        data = load_data(1000)

        st.sidebar.write('Unique Values')
        suggest = data['Department'].unique()
        st.write()


        # st.subheader('Project Location')
        # st.map(data)

    # ========= Add keys =========
    if uploaded_file is not None:
        st.subheader('Dictionary updater')
        unique = data["Department"].unique()
        st.sidebar.write(unique)
        # json_key = st.sidebar.selectbox("Enter key", unique) WE NEED TO MAKE THE FIRST RENDER BLANK SOMEHOW!!!!
        col1,col2,col3=st.columns(3)
        with col1:
            json_key = st.text_input("Enter key").lower()
        with col2:
           json_value = st.text_input("Enter value").lower()
        with col3:
            json_column = st.selectbox("Column to change", ['Department'])
        
        if st.button('🗸 Append to JSON'):
            # ========= json dictionary append try 1 =========

            def add_entry(name, element):
                    # return {name: {element: hashlib.md5(name.encode('utf-8')+element.encode('utf-8')).hexdigest()}}
                    return {name: element}

            #add entry
            entry = add_entry(json_key, json_value)

            #Update to JSONr
            with open('elements.json', 'r') as f:
                json_data = json.load(f)
                print(json_data.values()) # View Previous entries
                json_data.update(entry)

            with open('elements.json', 'w') as f:
                f.write(json.dumps(json_data))
        col4,col5,col6=st.columns([3.5,5.5,1])
        with col4:
            if st.button('✖ Delete key-value pair'):
                # ========= json dictionary delete try 1 =========
                #Update to JSONr
                with open('elements.json', 'r') as f:
                    dic=json.load(f)
                    print(dic.values())
                    try:
                        if dic[json_key]:
                            del dic[json_key]
                    except KeyError:
                        with col5:
                            st.warning('⚠️ Warning: Key doesn\'t exist!')           
                    with open('elements.json', 'w') as f:
                        f.write(json.dumps(dic))
                    with col6:
                        st.markdown('`IDLE`')


    # ========= json dictionary replace and display =========
    if uploaded_file is not None:
        st.subheader('JSON Replace')
        with open('elements.json', 'r') as JSON:
            json_dict = json.load(JSON)
        json_replaced = data.replace({"Department": json_dict})
        st.write(json_replaced)

    if st.button('Render and append to Master'):
        st.info('File downloaded. Appending to master has been manually disabled for demo purposes.')

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         * {font-family: "Balto";}
#         h1, h2, h3, h4, h5, h6 {font-family: "Balto";}
#         /* df cells header row face */
# 		.css-sc0g0 {font-family: "Balto";}
#         /* df cells font face */
#         .css-1l40rdr {font-family: "Balto";}
#         .st-af {font-size: 0.9rem;}
#         footer {visibility: hidden;}
#         footer:after {content:'Developed with <3 by Ansh Sharma at FivD'; visibility: visible; display: block; position: relative; padding: 5px; top: 2px;}
#         th {text-align:left;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        * {font-family: "Balto";}
        h1, h2, h3, h4, h5, h6 {font-family: "Balto";}
        /* df cells header row face */
      .css-sc0g0 {font-family: "Balto";}
        /* df cells font face */
        .css-1l40rdr {font-family: "Balto";}
        .st-af {font-size: 0.9rem;}
        footer {visibility: hidden;}
        footer:after {content:'Developed with <3 by Ansh Sharma at FivD'; visibility: visible; display: block; position: relative; padding: 5px; top: 2px;}
        th {text-align:left;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)