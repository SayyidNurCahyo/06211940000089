#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


data=pd.read_csv("C:/Users/LENOVO/OneDrive/Dokumen/andat/BankChurners.csv")
data


# In[3]:


np.sum(data.isna())


# In[4]:


data.info()


# In[5]:


def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal


# In[6]:


pal_vi = get_color('viridis_r', len(data['Attrition_Flag'].unique()))
pal_plas = get_color('plasma_r', len(data['Gender'].unique()))
pal_spec = get_color('Spectral', len(data['Income_Category'].unique()))
pal_hsv = get_color('hsv', len(data['Gender'].unique()))
pal_bwr = get_color('bwr', len(data['Marital_Status'].unique()))
pal_ = list(sns.color_palette(palette='plasma_r',
                              n_colors=len(data['Education_Level'].unique())).as_hex())


# In[7]:


fig = px.pie(data, values=data['Education_Level'].value_counts()[data['Education_Level'].unique()], names=data['Education_Level'].unique(),
             color_discrete_sequence=pal_)
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[8]:


fig = px.pie(data, values=data['Attrition_Flag'].value_counts()[data['Attrition_Flag'].unique()], names=data['Attrition_Flag'].unique(),
             color_discrete_sequence=pal_vi)
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[9]:


fig = px.pie(data, values=data['Gender'].value_counts()[data['Gender'].unique()], names=data['Gender'].unique(),
             color_discrete_sequence=pal_plas)
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[10]:


fig = px.histogram(data, x=data['Customer_Age'], color_discrete_sequence=pal_hsv,color='Gender',barmode='group')
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[11]:


a=['Less than 40K','40K - 60K','80K - 120K','60K - 80K','Unknown','120K+']
a


# In[12]:


fig = px.line_polar(data, r= data['Income_Category'].value_counts(),
                    theta=a, line_close=True)
fig.update_traces(fill='toself', line = dict(color=pal_spec[5]))
fig.show()


# In[13]:



sns.set_style('darkgrid')
g = sns.barplot(data=data, x=sorted(data['Dependent_count'].unique()), y=data['Dependent_count'].value_counts()[sorted(data['Dependent_count'].unique())],
                ci=False, palette='viridis_r')
g.set_xticklabels(sorted(data['Dependent_count'].unique()), rotation=45, fontdict={'fontsize':13})
plt.show()


# In[14]:


fig = px.pie(data, values=data['Marital_Status'].value_counts()[data['Marital_Status'].unique()], names=data['Marital_Status'].unique(),
             color_discrete_sequence=pal_bwr)
fig.update_traces(textposition='outside', textinfo='percent+label', 
                  hole=.6, hoverinfo="label+percent+name")
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.show()


# In[15]:


fig = px.scatter(data, x='Total_Amt_Chng_Q4_Q1', y='Total_Ct_Chng_Q4_Q1', color="Attrition_Flag")
fig.show()


# In[16]:


fig = px.scatter(data, x='Total_Trans_Amt', y='Total_Trans_Ct', color="Attrition_Flag")
fig.show()


# In[17]:


data=data.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'],axis=1)
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(round(data.corr(),3), annot=True, linewidths=0.5, fmt='.3f',ax=ax)
plt.show()


# In[ ]:





# In[ ]:




