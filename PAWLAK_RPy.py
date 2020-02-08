#!/usr/bin/env python
# coding: utf-8

# # Projet Rpy

# ## Python importation

# In[49]:


#!/usr/bin/env python3
# -*- coding: ISO-8859-1 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics
	
from sklearn.utils import resample


# ## Rpy2 importation

# In[50]:


from rpy2 import robjects
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import Formula, Environment
from rpy2.robjects.vectors import IntVector, FloatVector, StrVector
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr, data
from rpy2.rinterface_lib.embedded import RRuntimeError
from functools import partial
from rpy2.ipython import html
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter
import warnings

# R package names
packnames = ('ggplot2', 'stats','grDevices','readr','knitr','ggpllot2','dplyr','tidyr','questionr')

utils = importr('utils')
# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# Recoding R function which make them available in Python

rprint = robjects.globalenv.find("print")
stats = importr('stats')
grdevices = importr('grDevices')
base = importr('base')
summarytools = importr('summarytools')
readr = importr('readr')
knitr = importr('knitr')
ggplot2 = importr('ggplot2')
dplyr = importr('dplyr')
tidyr = importr('tidyr')
questionr = importr('questionr')
nortest = importr('nortest')

# For notebook
html.html_rdataframe=partial(html.html_rdataframe, table_class="docutils")


# ## Dataset loading

# In[51]:


pd.set_option('display.max_columns', None)
path = "/home/jofriii/Documents/M1_DSS/Rpy/Rea/export_data.csv"
data = pd.read_csv(path, delimiter=";", encoding="iso-8859-1", low_memory=False, decimal=',')


# ## Exploratory analysis

# ### Selecting data of interrest and visualisation

# In[52]:


stat_descriptives = data.iloc[:,0:16]
stat_descriptives['DUREE_SEJOUR'] = data['DUREE_SEJOUR_INTERV'].values
stat_descriptives = stat_descriptives.drop(['ID_INTERVENTION',
                                              'ID_PATIENT',
                                              'POIDS_IDEAL_LORENTZ',
                                              'POIDS_IDEAL',
                                              'DELTA_POIDS_POIDS_IDEAL',
                                             'URGENCE',
                                           "CATEGORIE_AGE_ADULTE"], axis = 1)
stat_descriptives = stat_descriptives[stat_descriptives.SEXE != "I"]
stat_descriptives = stat_descriptives[stat_descriptives.AGE >18]
stat_descriptives = stat_descriptives[stat_descriptives.POIDS < 400]
cut_label = ['moins de 18 ans','18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','plus de 75 ans']
cut_bins = [18,20,25,30,35,40,45,50,55,60,65,70,75,150]
stat_descriptives['CATEGORIE_AGE'] = pd.cut(stat_descriptives.AGE, bins=cut_bins, labels=cut_label, right = False)


var=['AGE','POIDS','TAILLE','IMC','DUREE_SEJOUR']
for v in stat_descriptives[var]:
    sns.set()
    chart = sns.distplot(stat_descriptives[v])
    chart = plt.xlabel(v)
    chart = plt.ylabel('Pourcentage')
    chart = plt.title("Répartition au sein de l'échantillon")
    plt.show()
    
    sns.set()
    chart = sns.boxplot(x=stat_descriptives.CATEGORIE_AGE, y=stat_descriptives[v])
    chart = plt.xlabel('AGE')
    chart = plt.ylabel(v)
    chart = plt.title("Boxplot en fonction de la classe d'âge")
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light')
    plt.show()
        


# In[53]:


### Détection des valeurs aberrantes


# ## Outliers management

# In[54]:


##robjects.r("outlier_values <- boxplot.stats(inputData$pressure_height)$out")
pandas2ri.activate()
var=['AGE','POIDS','TAILLE','IMC','DUREE_SEJOUR']
for i in stat_descriptives[var]:
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_summary = grdevices.boxplot_stats(stat_descriptives[i])
        
    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_from_r_df = np.asarray(ro.conversion.rpy2py(df_summary))
        print(i,pd_from_r_df[0])
pandas2ri.deactivate()


# In[55]:


data_poids = stat_descriptives['POIDS'] < 119
data_poids2 = stat_descriptives['POIDS'] > 28
data_age = stat_descriptives['AGE'] < 106
data_age2 = stat_descriptives['AGE'] > 19
data_taille = stat_descriptives['TAILLE'] < 194
data_taille2 = stat_descriptives['TAILLE'] > 143
data_imc = stat_descriptives['IMC'] < 40
data_imc2 = stat_descriptives['IMC'] > 11.7
data_sejour = stat_descriptives['DUREE_SEJOUR'] < 16
data_select = stat_descriptives[data_age & data_taille2 & data_taille & data_poids & data_age2 & data_poids2 & data_imc & data_imc2 & data_sejour]


# In[56]:


var=['AGE','POIDS','TAILLE','IMC','DUREE_SEJOUR']
for v in data_select[var]:
    sns.set()
    chart = sns.distplot(data_select[v])
    chart = plt.xlabel(v)
    chart = plt.ylabel('Pourcentage')
    chart = plt.title("Répartition au sein de l'échantillon")
    plt.show()
    
    sns.set()
    chart = sns.boxplot(y=data_select[v])
    chart = plt.ylabel(v)
    chart = plt.title(v)
    plt.show()


# ## Statistical analysis

# ### Quantitative variables

# In[57]:


var=['AGE','POIDS','TAILLE','IMC', 'DUREE_SEJOUR']
for v in [var] :
    stat_descriptives_r = data_select.loc[:,v]
    
    pandas2ri.activate()
    with localconverter(ro.default_converter + pandas2ri.converter):
      df_summary = base.summary(stat_descriptives_r)    
    print(v,df_summary)

    
print(statistics.stdev(data_select.AGE))
print(statistics.stdev(data_select.POIDS))
print(statistics.stdev(data_select.TAILLE))
print(statistics.stdev(data_select.IMC))


# In[58]:


sns.set()
chart = sns.boxplot(x=data_select.SERVICE, y=data_select['DUREE_SEJOUR'])
chart = plt.xlabel('AGE')
chart = plt.ylabel(v)
chart = plt.title("Boxplot en fonction de la classe d'âge")
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light')
plt.show()


# ### Qualitative variables

# In[59]:


vqList=['CATEGORIE_AGE','CATEGORIE_IMC_ADULTE','ASA', 'CATEGORIE_ASA','SEXE']
for vq in data_select[vqList]:
    sns.countplot(data_select[vq])
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light')
    plt.show()


# In[60]:


vqList=['CATEGORIE_AGE','CATEGORIE_IMC_ADULTE','ASA', 'CATEGORIE_ASA','SEXE']
data_qual = data_select.loc[:,vqList]
data_qual.info()

data_qual.ASA = data_qual.ASA.astype('category')
data_qual.CATEGORIE_ASA = data_qual.CATEGORIE_ASA.astype('category')
data_qual.SEXE = data_qual.SEXE.astype('category')
data_qual.CATEGORIE_AGE = data_qual.CATEGORIE_AGE.astype('category')
data_qual.CATEGORIE_IMC_ADULTE = data_qual.CATEGORIE_IMC_ADULTE.astype('category')

for vq in data_select[vqList]:
    data_sel=data_qual[vq]
    pandas2ri.activate()
    with localconverter(ro.default_converter + pandas2ri.converter):
      df_summary = questionr.freq(data_sel)
    print('\n',vq,'\n',df_summary)
    pandas2ri.deactivate()
    


# ## Inferential analysis

# ### ANOVA

# #### Subset principal set into two group : ASA-low and ASA-high

# In[61]:


data_ASAinf = data_select.loc[data_select['CATEGORIE_ASA'] == 'ASA1-2']
print(data_ASAinf.shape) # Vérification du nombre correct de ligne
print(data_ASAinf.columns)

data_ASAsup = data_select.loc[data_select['CATEGORIE_ASA'] == 'ASA3-4-5']
print(data_ASAsup.shape) # Vérification du nombre correct de ligne
print(data_ASAsup.columns)


# #### Unbalanced datasets, downgradation of the highest one

# In[62]:


data_ASAsup = resample(data_ASAsup,
                       replace=False,
                       n_samples=33530,
                       random_state=123) 
print(data_ASAsup.shape)


# #### Normality test on explained variable

# In[63]:


with localconverter(ro.default_converter + pandas2ri.converter):
    modele = nortest.lillie_test(data_ASAinf["DUREE_SEJOUR"])
with localconverter(ro.default_converter + pandas2ri.converter):
    pd_from_r_df = np.asarray(ro.conversion.rpy2py(modele))
    print("\n Normality test on sample with ASA 1 and 2 \n")
    print('D=',pd_from_r_df[0])
    print('p_value =', pd_from_r_df[1], '* \n * : < 2.2E-16')


# In[64]:


pandas2ri.deactivate()

with localconverter(ro.default_converter + pandas2ri.converter):
    modele = nortest.lillie_test(data_ASAsup["DUREE_SEJOUR"])
with localconverter(ro.default_converter + pandas2ri.converter):
    pd_from_r_df = np.asarray(ro.conversion.rpy2py(modele))
    print("\n Normality test on sample with ASA 3, 4 and 5 \n")
    print('D=',pd_from_r_df[0])
    print('p_value =', pd_from_r_df[1], '* \n * : < à 2.2E-16')
pandas2ri.deactivate()

print("\n\n Samples are not normality distributed \n")


# ###### Wilcoxon test on mean comparison between sex no matter ASA groups

# In[65]:


data_M = data_select.loc[data_select['SEXE'] == 'M']
print(data_M.shape) # Vérification du nombre correct de ligne
print(data_M.columns)

data_F = data_select.loc[data_select['SEXE'] == 'F']
print(data_F.shape) # Vérification du nombre correct de ligne
print(data_F.columns)

with localconverter(ro.default_converter + pandas2ri.converter):
    modele = stats.wilcox_test(data_M["DUREE_SEJOUR"],data_F["DUREE_SEJOUR"])
with localconverter(ro.default_converter + pandas2ri.converter):
    pd_from_r_df = np.asarray(ro.conversion.rpy2py(modele))
    print('D=',pd_from_r_df[0])
    print('p_value =', pd_from_r_df[2], '* \n * : < à 2.2E-16')
pandas2ri.deactivate()

print(data_M.DUREE_SEJOUR.mean())
print(data_F.DUREE_SEJOUR.mean())


# ##### Khi-2 test on ASA 3-4-5 category

# In[66]:


var_anova_1=['SEXE','CATEGORIE_IMC_ADULTE', 'CATEGORIE_AGE', ]
for i in var_anova_1:
    sns.boxplot(y=data_ASAsup.DUREE_SEJOUR, x=data_ASAsup[i])
    plt.show()


# In[67]:


vqList=['SEXE','CATEGORIE_IMC_ADULTE']
for i in vqList:
        col = ["CATEGORIE_AGE",i]
        print(col)
        table_khi2 = data_ASAsup[col]
        tab_cont = table_khi2.pivot_table(index = 'CATEGORIE_AGE', columns = i ,aggfunc =len)
        print('\n\n\n ASA3-4-5 category \n\n\n Contigency table : \n\n',tab_cont)
        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
          res_chi2 = stats.chisq_test(tab_cont)
        print(res_chi2)
        pandas2ri.deactivate()


# In[68]:


vqList=['CATEGORIE_AGE','CATEGORIE_IMC_ADULTE']
for i in vqList:
        col = ["SEXE",i]
        print(col)
        table_khi2 = data_ASAsup[col]
        tab_cont = table_khi2.pivot_table(index = 'SEXE', columns = i ,aggfunc =len)
        print('\n\n\n ASA3-4-5 Category \n\n\n Contigency table : \n\n',tab_cont)
        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
          res_chi2 = stats.chisq_test(tab_cont)
        print(res_chi2)
        pandas2ri.deactivate()


# In[69]:


vqList=['CATEGORIE_AGE','SEXE']
for i in vqList:
        col = ['CATEGORIE_IMC_ADULTE',i]
        table_khi2 = data_ASAsup[col]
        print(table_khi2)
        tab_cont = table_khi2.pivot_table(index = 'CATEGORIE_IMC_ADULTE', columns = i ,aggfunc =len)
        print('\n\n\n ASA3-4-5 category \n\n\n Contigency table : \n\n',tab_cont)
        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
          res_chi2 = stats.chisq_test(tab_cont)
        print(res_chi2)
        pandas2ri.deactivate()


# ###### Wilcoxon test for mean comparison based on sex within ASA 3-4-5 group

# In[70]:


data_M = data_ASAsup.loc[data_select['SEXE'] == 'M']
print(data_M.shape) # Vérification du nombre correct de ligne
print(data_M.columns)

data_F = data_ASAsup.loc[data_select['SEXE'] == 'F']
print(data_F.shape) # Vérification du nombre correct de ligne
print(data_F.columns)

with localconverter(ro.default_converter + pandas2ri.converter):
    modele = stats.wilcox_test(data_M["DUREE_SEJOUR"],data_F["DUREE_SEJOUR"])
print(res_chi2)
pandas2ri.deactivate()

print(data_M.DUREE_SEJOUR.mean())
print(data_F.DUREE_SEJOUR.mean())


# ##### Khi-2 test on ASA 1-2 category

# In[71]:


vqList=['SEXE','CATEGORIE_IMC_ADULTE']
for i in vqList:
        col = ["CATEGORIE_AGE",i]
        print(col)
        table_khi2 = data_ASAinf[col]
        tab_cont = table_khi2.pivot_table(index = 'CATEGORIE_AGE', columns = i ,aggfunc =len)
        print('\n\n\n Catégorie ASA 1 et 2 \n\n\n Contigency table : \n\n',tab_cont)
        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
          res_chi2 = stats.chisq_test(tab_cont)
        print(res_chi2)
        pandas2ri.deactivate()


# In[72]:


vqList=['CATEGORIE_AGE','CATEGORIE_IMC_ADULTE']
for i in vqList:
        col = ["SEXE",i]
        print(col)
        table_khi2 = data_ASAinf[col]
        tab_cont = table_khi2.pivot_table(index = 'SEXE', columns = i ,aggfunc =len)
        print('\n\n\n Catégorie ASA 1 et 2 \n\n\n Contigency table : \n\n',tab_cont)
        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
          res_chi2 = stats.chisq_test(tab_cont)
        print(res_chi2)
        pandas2ri.deactivate()


# In[73]:


vqList=['CATEGORIE_AGE','SEXE']
for i in vqList:
        col = ['CATEGORIE_IMC_ADULTE',i]
        table_khi2 = data_ASAinf[col]
        print(table_khi2)
        tab_cont = table_khi2.pivot_table(index = 'CATEGORIE_IMC_ADULTE', columns = i ,aggfunc =len)
        print('\n\n\n Catégorie ASA 1 et 2 \n\n\n Contigency table : \n\n',tab_cont)
        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
          res_chi2 = stats.chisq_test(tab_cont)
        print(res_chi2)
        pandas2ri.deactivate()


# ###### Wilcoxon test for mean comparison based on sex within ASA 1-2 group

# In[74]:


data_M = data_ASAinf.loc[data_select['SEXE'] == 'M']
print(data_M.shape) # Vérification du nombre correct de ligne
print(data_M.columns)

data_F = data_ASAinf.loc[data_select['SEXE'] == 'F']
print(data_F.shape) # Vérification du nombre correct de ligne
print(data_F.columns)

with localconverter(ro.default_converter + pandas2ri.converter):
    modele = stats.wilcox_test(data_M["DUREE_SEJOUR"],data_F["DUREE_SEJOUR"])
print(res_chi2)
pandas2ri.deactivate()

print(data_M.DUREE_SEJOUR.mean())
print(data_F.DUREE_SEJOUR.mean())


# ### Wilcoxon test for mean comparison 

# In[75]:


data_sejour_ASAinf = data_ASAinf.loc[data_select['CATEGORIE_ASA'] == 'ASA1-2']
print(data_sejour_ASAinf.shape) # Vérification du nombre correct de ligne
print(data_sejour_ASAinf.columns)

data_sejour_ASAsup = data_ASAsup.loc[data_select['CATEGORIE_ASA'] == 'ASA3-4-5']
print(data_sejour_ASAsup.shape) # Vérification du nombre correct de ligne
print(data_sejour_ASAsup.columns)

with localconverter(ro.default_converter + pandas2ri.converter):
    modele = stats.wilcox_test(data_sejour_ASAinf["DUREE_SEJOUR"],data_sejour_ASAsup["DUREE_SEJOUR"])
print(res_chi2)
pandas2ri.deactivate()

print(data_sejour_ASAinf.DUREE_SEJOUR.mean())
print(data_sejour_ASAsup.DUREE_SEJOUR.mean())


# In[ ]:




