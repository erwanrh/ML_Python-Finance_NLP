#!/usr/bin/env python
# coding: utf-8

# # Projet Machine Learning : L'importance des mots
# Rahis Erwan

# Ce sujet a pour but de créer un algorithme permettant d'analyser les occurences de mots dans l'actualité liée à des cours de la bourse et de déterminer s'ils indiquent un signal d'achat ou de vente.


# %% Install libraries
#Si besoin
#!{sys.executable} -m pip install plotly
#!{sys.executable} -m pip install imblearn
#!{sys.executable} -m pip install scikit-learn
#!pip install pandoc


# %% Import 

import sys

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns

import matplotlib.pyplot as plt 
import plotly.graph_objects as plty
import plotly.express as px
import plotly.graph_objects as go

import math
import datetime
from datetime import datetime


# %% Data

#Lecture de la base
CAC40_data=pd.read_csv('Data/cac40_v3.csv', sep=",", decimal=".") #Chargement du fichier


#%% Caractéristiques de la base

CAC40_data = CAC40_data.drop(columns=['Unnamed: 0']) #supprime la colonne Unamed
CAC40_data.shape

# ### Nombre de lignes et colonnes
# La base possède : 
# 106 542 lignes 
# 280 colonnes (281 avec Unnamed)

# Nous ajoutons une colonne de date dans le format datetime afin d'avoir les dates en abscisse dans les graphiques.
CAC40_data['date']=pd.to_datetime({'year':CAC40_data['annee'],'month':CAC40_data['mois'], 'day':CAC40_data['jour']})
print(CAC40_data.head(5))


# %%  Statistiques de base

#  Rendements annuels par ticker
tickers_list = CAC40_data.TICKER.unique() 

#  Moyenne des rendements sur tout l'historique, par ticker
CAC40_TickerMeans= CAC40_data[['TICKER','annee','RDMT_M','RDMT_J','RDMT_S']].groupby(['TICKER','annee']).mean()
CAC40_TickerMeans.reset_index(inplace=True)  


#  Graphique des moyennes des rendements par ticker par année
fig = px.line(CAC40_TickerMeans, x='annee', y='RDMT_M', title='Yearly mean of daily returns for each ticker', color='TICKER')
fig.update_layout(xaxis_title="Year",yaxis_title="Daily Returns Mean")
fig.show()


#  Ecarts-types des rendements sur tout l'historique, par ticker
CAC40_TickerStd = CAC40_data[['TICKER','RDMT_M']].groupby('TICKER').std()
CAC40_TickerHistMean = CAC40_data[['TICKER','RDMT_M']].groupby('TICKER').mean()
CAC40_TickerPlot = pd.concat([CAC40_TickerStd, CAC40_TickerHistMean], axis=1)
CAC40_TickerPlot.columns = ['Std','Mean']
CAC40_TickerPlot.reset_index(inplace=True)  


#  Graphique des volatilités de chaque ticker sur l'ensemble de l'historique
fig = px.bar(CAC40_TickerPlot, x='TICKER', y='Std', title='Historical volatility of daily returns for each ticker')
fig.update_layout(xaxis_title="Year",yaxis_title="Daily Returns Volatility")
fig.show()


#  Graphique des moyennes des rendements de chaque ticker sur tout l'historique
fig = px.bar(CAC40_TickerPlot, x='TICKER', y='Mean', title='Historical mean of daily returns for each ticker')
fig.update_layout(xaxis_title="Year",yaxis_title="Daily Returns Mean")
fig.show()


#  Moyenne des rendements de tous les tickers sur tout l'historique
CAC_historicalMean_rdt = pd.DataFrame({'Returns':CAC40_data[['RDMT_M','RDMT_J','RDMT_S']].mean()*100}).reset_index().melt(id_vars='index')
CAC_historicalMean_prices = pd.DataFrame({'Prices':CAC40_data[['OP','UP','DO','CL']].mean()}).reset_index().melt(id_vars='index')
CAC_historicalMean_vol = pd.DataFrame({'Volume':CAC40_data[['VO']].mean()}).reset_index().melt(id_vars='index')

CAC_historicalMeans = pd.DataFrame.append(CAC_historicalMean_rdt,[CAC_historicalMean_prices,CAC_historicalMean_vol], ignore_index= True)


fig = px.bar(CAC_historicalMeans, x='index', y='value',color='index', title='Historical means of data ',
             facet_col="variable")
fig.update_yaxes(title = '',matches=None, showticklabels=True)
fig.update_xaxes(title = '',matches=None)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(showlegend=False)
fig.show()


#  Moyenne des rendements mensuels / journaliers / hebdomadaires  par année 

CAC40_YearMeans = CAC40_data[['annee','OP','UP','DO','CL','VO','RDMT_M','RDMT_J','RDMT_S']].groupby('annee').mean()
fig = px.line(CAC40_YearMeans, x=CAC40_YearMeans.index, y='RDMT_M', title='Yearly mean of monthly returns',)
fig.update_layout(xaxis_title="Year",yaxis_title="Daily Returns Mean")
fig.show()


#  Moyenne du nombre d'apparition des mots

print('Mots qui apparaissent le plus')
print(CAC40_data.iloc[:,25:].mean().sort_values(ascending=False).head(5))
print('\nMots qui apparaissent le moins')
print(CAC40_data.iloc[:,25:].mean().sort_values(ascending=False).tail(5))


CAC40_wordsMean = CAC40_data.iloc[:,25:].sum(axis=0)
CAC40_wordsMean=CAC40_wordsMean.reset_index()
CAC40_wordsMean.columns=['Word','NbApparition']

# Les mots les plus utilisés sont pour, avec, sous, pres, dans. 

#  Pie chart des mots qui apparaissent plus de 400 fois

fig = px.pie(CAC40_wordsMean[CAC40_wordsMean['NbApparition']>400], names='Word', values='NbApparition', title='Mots les plus utilisés')
fig.update_traces(textposition='inside', textinfo='label')
fig.update_layout(showlegend=False, title_x=0.5)
fig.show()


# %%  Mots pertinents
# Il y a 23 colonnes de caractéristiques des actions donc les colonnes contenantles mots commencent à la colonne 24

CAC40_wordsonly=CAC40_data.iloc[:,24:-1]
CAC40_wordsonly.head()


#  Tableau des mots apparus plus de 400 fois et dont le rendement mensuel moyen du ticker est >1%

relevant_words = pd.DataFrame(columns=['Word_Nb','RDMT_M']) 

for col in CAC40_wordsonly.columns:
        if (CAC40_wordsonly.loc[:,col].sum()>400) and (CAC40_data[CAC40_data[col]==1].RDMT_M.mean()>0.01):
            relevant_words.loc[col,]=[CAC40_wordsonly.loc[:,col].sum(),CAC40_data[CAC40_data[col]==1].RDMT_M.mean()]


print(relevant_words)


# Les mots de cette liste sont liés à des rendements supérieurs à 1%. Ils peuvent être utilisés pour prévoir un bon rendement et ainsi aider à la décision d'achat ou de vente.

#  Lignes contenant au moins un des mots ci-dessus

CAC40_rowsclean_wordsonly = CAC40_wordsonly[relevant_words.index].loc[CAC40_wordsonly[relevant_words.index].sum(axis=1)>=1]
CAC40_rowsclean = CAC40_data.iloc[:,:23][CAC40_data[relevant_words.index].sum(axis=1)>=1].join(CAC40_rowsclean_wordsonly)
CAC40_rowsclean.head()


# Les tableaux ci-dessus contiennent seulement les lignes ayant un mot apparaissant au moins une fois. Le premier tableau contient toutes les colonnes et le deuxième contient seulement les colonnes de mots. 

#%%: Retrait des variables trop corélées
#  Corrélogramme
# 1. Matrice de corrélation

corr_matrix = CAC40_wordsonly[relevant_words.index].corr()
corr_matrix100 = corr_matrix * 100


# 2. Heatmap de la matrice de corrélation

fig = go.Figure(data=go.Heatmap({'z': corr_matrix100.values.tolist(),
            'x': corr_matrix.columns.tolist(),
            'y': corr_matrix.index.tolist()}, 
               colorscale = [[0, '#E582CB'], [0.5, '#F9E79F'],[1, '#D7BDE2']],
                               zmin=-100, zmax=100))
fig.update_xaxes(side="top")
fig.update_layout(title="Words correlation matrix",
    xaxis_title="", yaxis_title="Words", xaxis_autorange = "reversed", 
    width=750, height=750, autosize=False)
fig.show()


# Nous pouvons observer sur le corrélogramme que nous avons des variables trop corrélées. Quand nous regardons de plus près, nous remarquons que la liste contient des mots similaires tels que 'nouvel' ou 'nouvelle', 'capital.' ou 'capital', 'group' ou 'groupe'. Ces variables sont trop similaires il faut les retirer.

#  Retrait des variables
# Nous choisissons de retirer toutes les variables qui ont une corrélation supérieure à 0.75. 

varcorr_ecartees = []
nb_variables = corr_matrix.shape[1]
for i in range(1,nb_variables):
    for j in range(i):
        if corr_matrix.iloc[i,j]>0.75:
            print(corr_matrix.columns[i]+"-"+corr_matrix.columns[j])
            if(corr_matrix.columns[i]!="RMDT_S" and corr_matrix.columns[j]!="RMDT_S"):
                varcorr_ecartees = varcorr_ecartees+[corr_matrix.columns[i]]


# Liste de variables écartées

varcorr_ecartees = list(set(varcorr_ecartees))
varcorr_ecartees

CAC40_rowsclean_nocorr = CAC40_rowsclean.drop(columns=varcorr_ecartees)
CAC40_clean_wordsonly_nocorr= CAC40_rowsclean_wordsonly.drop(columns=varcorr_ecartees)


# Les deux tableaux ci-dessus contiennent notre base, l'un avec toutes les colonnes, l'autre avec seulement les colonnes de mots. 

#%% Algorithme 

# Nous choisissons l'algorithme Multi-layer Perceptron classifier qui fait partie de la famille des réseaux de neurones. 

# ### Import des packages

# In[27]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE


# ### Préparation des données
# Nous utilisons en variable à expliquer les rendements mensuels et en variables explicatives les mots après filtrage de la base. 

# In[28]:


y_full = CAC40_rowsclean_nocorr['RDMT_M']
X_full = CAC40_clean_wordsonly_nocorr


# Le tableau ci-dessus contient seulement les mots pertinents qui ne sont pas trop corrélés.

# Sampley est le tableau contenant les rendements mensuels du CAC40 dont les lignes contiennent au moins un 1.
# SampleX est le tableau contenant ces mêmes lignes mais seulement les colonnes des mots. 

# https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

# ## Découpage de la base,  normalisation et binarasiation de chaque partie des rendements
# Dans un premier temps nous testons notre algorithme avec 10% de la base afin de trouver une configuration convenable avant de le lancer sur la base complète. 

# #### Test de l'agorithme sur 10% de la base

# Nous prenons 10 pourcent de notre base en créant un array d'entiers aléatoires entre 0 et la taille de la base. Ces nombres aléatoires déterminerons quelles lignes vont consituter la base de 10%. 

# In[29]:


size_base = len(y_full)
indeX = np.random.randint(0, size_base-1, round(0.1*size_base))


# Nous créeons la base de 10%. Nous avons fait nos premiers tests avec une base de 10% pour élaguer les paramètres et définir des intervalles de variation.

# In[30]:


y = y_full#.iloc[indeX]
X = X_full#.iloc[]


# ### Création des bases entrainement et test
# Nous découpons la base entre une base d'entrainement et une base de test. 

# In[94]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)


# #### Normalisation des rendements
# Nous faisons des essais avec les rendements de la base test et de la base train normalisés. 

# In[95]:


y_train_norm = pd.DataFrame(sklearn.preprocessing.normalize(y_train.values.reshape(-1,1), axis=0))
y_test_norm  = pd.DataFrame(sklearn.preprocessing.normalize(y_test.values.reshape(-1,1), axis=0))
y_train_norm.head()


# ### Binarisation des rendements
# Nous binarisons les rendements afin d'obtenir un indicateur dans le cas où les rendements sont supérieurs à 2%.
# Si le rendement est >2% alors la variable binaire prend la valeur 1, et 0 sinon. Nous choisissons les rendements mensuels pour effectuer l'algorithme.

# In[96]:


y_train_norm_label = pd.DataFrame({'y':[1 if v > 0.02 else 0 for v in y_train_norm[0]]}) 
y_test_norm_label = pd.DataFrame({'y':[1 if v > 0.02 else 0 for v in y_test_norm[0]]}) 


# In[97]:


y_train_label = pd.DataFrame({'y':[1 if v > 0.02 else 0 for v in y_train]}) 
y_test_label = pd.DataFrame({'y':[1 if v > 0.02 else 0 for v in y_test]}) 


# In[98]:


y_train_norm_label.head()


# #### Graphique des rendements binaires

# In[58]:


#Données
plot_ylabel = pd.DataFrame()
plot_ylabel = plot_ylabel.append({'y':0, 'count': y_train_norm_label[y_train_norm_label['y']==0].count()['y'], 'norm':'YES' }, ignore_index=True)
plot_ylabel = plot_ylabel.append({'y':1, 'count': y_train_norm_label[y_train_norm_label['y']==1].count()['y'], 'norm':'YES' }, ignore_index=True)
plot_ylabel = plot_ylabel.append({'y':0, 'count': y_train_label[y_train_label['y']==0].count()['y'], 'norm':'NO' }, ignore_index=True)
plot_ylabel = plot_ylabel.append({'y':1, 'count': y_train_label[y_train_label['y']==1].count()['y'], 'norm':'NO' }, ignore_index=True)


# In[59]:


fig = px.bar(plot_ylabel, x='count', y='y',color='norm',barmode='group', orientation='h',title = 'Returns as dummy variable count')
fig.update(layout_yaxis_dtick = 1, layout_xaxis_title="Count", layout_yaxis_title="Return Label (1:>2%, 0:<=2% )")
fig.show()


# Graphique des rendements binarisés normalisés en bleu et non normalisés en rouge. Nous pouvons observer qu'en normalisant les rendements (nous regardons ici seulement la base d'entrainement), la proportion de rendements supérieurs à 2% est plus grande. 
# Initialement, nous normalisons pour réduire la compléxité du modèle en réduisant les dégrés de liberté qui concernent la moyenne et l'écart-type.
# Nous verrons avec les résulats si la normalisation joue un rôle dans les résultats. 
# Nous nous attendons à avoir de meilleurs résultats avec la base non normalisée sachant qu'il y a plus de 1. 

# #### Graphique de l'apparition des mots en fonction des rendements

# In[60]:


plot_words_y = y_train_label.join(X_train.reset_index(drop=True))
plot_words_y = plot_words_y.groupby('y').sum().apply(lambda x: 100 * x / float(x.sum()),axis=1).reset_index().melt(id_vars= 'y')


# In[61]:


fig = px.bar(plot_words_y, x='value', y='y',color='variable', orientation='h', title = 'Words')
fig.update(layout_yaxis_dtick = 1,layout_xaxis_title="Percentage of word occurence", layout_yaxis_title="Return Label (1:>2%, 0:<=2% )")
fig.show()


# Ce graphique montre le pourcentage d'apparition des mots sachant que le rendements est >2% ou <=2%. Les deux barres montrent un pattern similaire. Nous ne pouvons pas déterminer si un mot est plutôt associé à des bons ou mauvais rendemement de cette manière. Les rendements sont répartis de la même manière dans les rendements élevés ou non.  Nous pensons tout de même à effectuer cette représentation en 3 dimensions en regardant des couples de mots.

# #### Heatmap des couples de mots et des "bons" rendements associés

# In[62]:


#Base training complète avec rendements binarisés
base_train_labeled = y_train_label.join(X_train.reset_index(drop=True))

plot_3D_words = pd.DataFrame()
for w1 in X_train.columns:
    for w2 in X_train.columns:
        if base_train_labeled[base_train_labeled.loc[:,[w1,w2]].sum(axis=1) == 2].shape[0] != 0 :
            plot_3D_words.loc[w1,w2] = 100*base_train_labeled[base_train_labeled.loc[:,[w1,w2,'y']].sum(axis=1) == 3].shape[0]/base_train_labeled[base_train_labeled.loc[:,[w1,w2]].sum(axis=1) == 2].shape[0]
        else:
            plot_3D_words.loc[w1,w2] = 0


# In[63]:


#Heatmap Plotly
fig = go.Figure(data=go.Heatmap({'z': plot_3D_words.values.tolist(),
            'x': plot_3D_words.columns.tolist(),
            'y': plot_3D_words.index.tolist()}, 
               colorscale = [[0, '#FEF9E7'], [1, '#52BE80']], 
                               zmin=0, zmax=100))
fig.update_xaxes(side="top")
fig.update_layout(title="Percentage of returns >2% by word couples",
    xaxis_title="",yaxis_title="Words",xaxis_autorange = "reversed",
    width=700, height=700, autosize=False)
fig.show()


# Ce graphique permet déjà d'observer des comportements liés à l'apparition de deux mots simultanément. Par exemple dans 100% des cas où les mots 'contrat' et 'titre' apparaissent simultanément, les rendements sont supérieurs à 2%. Le réseau de neurones va donc capter ces informations dans des dimensions encore plus élevées. Cette heatmap n'est qu'une représentation améliorée du Bar Plot fait plus haut mais ne nous permet pas de tirer de vraie conclusion quant à la relation entre l'apparition des mots et le taux de rendement. 

# ## Création de l'algorithme
# 
# Nous choisissons l'algorithme MLP : MultiLayer Perceptron qui fait partie de la famille des réseaux de neurones. L'idée générale est d'imiter le fonctionnement d'un neurone humain en sommant les informations (signaux) reçues et provenant d'autres neurones, en traitant cette somme d'informations et en la transmettant aux autres neurones. Le renforcement des liens entre les neurones conduit à l'apprentissage.
# C'est un modèle qui est très précis et performant sur de grandes bases. Les difficultées viennent du paramétrage difficile, le choix du nombre de neurones de la couche cachée, le sur-apprentissage et le problème de non convergence vers l'optimum. Mathématiquement le modèle optimise une fonction de log-pertes en utilisant un type de solver choisi en paramètre. 

# ### Function pour avoir les métriques.
# Cette fonction va nous permettre de remplir un Dataframe contenant tous les paramètres et les caractéristiques des modèles ainsi que les métriques associées pour les classer in fine.
# 

# In[42]:


def model_metrics(type_model, y_cv, predictions, normalisation, smot, solver, fold, activation, hl, alpha,
                 b1 = 'NaN', b2 = 'NaN', ep='NaN'):
  
    auc = round(roc_auc_score(y_cv, predictions),4)
    precision = round(precision_score(y_cv, predictions, zero_division= 0),4)
    recall = round(recall_score(y_cv, predictions),4)
    
    return([{'00_type_model' : type_model,
             '0_auc' : auc,
             '0_precision' : precision,
             '0_recall' : recall, 
             '1_normalisation' :normalisation, 
             '1_SMOT':smot, 
             '1_fold': fold,
             
             '2_solver': solver,
             
             '3_activation': activation,
             '3_hiddenlayer': hl,
             '3_alpha': alpha,

             
             '4_beta1ADAM': b1,
             '4_beta2ADAM': b2,
             '4_epsilonADAM': ep
           }])


# ### Hyperparameters & Cross Validation & Overfitting
# Nous voulons trouver les meilleurs hyperparamètres du modèle pour affiner les prédiction.

# Dans un premier temps nous avons effectué une centaine de fits des trois types de solver. Nous continuons avec les solvers adam et lbfgs. En effet, le tableau de métriques du premier run testant 181 modèles avec des solver et des paramètres différents met en avant les performances moindres du solver sgd pourles trois métriques. Le sgd est un solver signifiant Stochastique Gradient Descent, il est plus performant et rapide pour des petites bases de données. Le solver Adam est un algorithme d'optimisation qui est également un gradient stochastique et permet d'avoir des résultats fiables et rapides. Le solver lbfgs est un algorithme de la famille des méthodes quasi-Newton qui utilise une quantité de mémoire limitée. 
# 
# 
# Dans un second temps, nous avons fitté 3745 modèles avec des paramètres différents afin d'afiner et de filtrer encore les hyperparamètres. Un tableau croisé dynamique des paramètres et des résultats nous a permis de déduire les principaux paramètres les plus adaptés à notre modèle. Les résultats sur données normalisées et avec oversampling se montrent meilleurs.
# 
# 
# Nous avons donc affiné le grid search en retirant les paramètres qui ne semblaient pas efficaces. Nous avons donc gardé trois Hidden Layers et trois alphas. Les Hidden Layers représentent neurones dans chaque couches cachées. Nous choisissons de n'utiliser qu'une seule couche. A chaque fois que l'on ajoute une couche, cela augmente la capacité d'abstraction du modèle (le modèle est capable de prendre en compte des corrélations à plusieurs niveau), en revanche lorsque l'on augmente le nombre de couches et le nombre de paramètres, nous avons besoin d'une plus grande base d'entrainement. Dans un souci de parcimonie et étant donné que nous avons 7 000 samples, nous ne testerons pas de réseaux à plusieurs couches. Dans la plupart des réseaux de neurones observés, la première couche possède autant de neurones que d'inputs. C'est pour cela que nous avons choisi 10 et 20 comme paramètres. Nous avons rajouté 50 car nous avons constaté qu'une augmentation du nombre de neurones pouvait améliorer l'efficacité du modèle.   
# 
# Les fonctions d'activations des couches cachées sont des équations mathématiques qui déterminent si le neurone doit être activé ou non selon l'information entrant dans le neurone. La fonction permet aussi de normaliser les informations en leur donnant un résultat entre 0 et 1. Nous sélectionnons trois fonctions d'activation : relu, identity et logistic. La fonction relu est une fonction binaire : $f(x)=max(0,x)$, la fonction identity est la fonction $f(x)=x$ et la fonction logistic est la fonction $f(x)=\frac{1}{1+exp(-x)}$. Nous n'avons pas retenu la fonction tanh car elle est ressemblante à la fonction logistique. 
# 
# Le paramètre alpha est un paramètre de pénalité. Il sert à régulariser la fonction objectif. Dans cet algorithme la fonction objectif est la fonction log-perte du modèle (mesure de la distance entre les valeurs observées et les valeurs calculées). Ajouter une pénalité signifie mutliplier le deuxième terme de la fonction objectif par cette pénalité pour ajuster le résultat. Un alpha élevé va réduire la variance qui est un signe d'overfitting. A l'inverse un alpha plus faible va réduire le biai, signe d'un underfitting. Nous choisissons trois valeurs pour alpha : 0.1, 0.01, 0.2, filtrées après plusieurs tentatives.Le paramètre alpha est un paramètre de pénalité. Il sert à régulariser la fonction objectif. Dans cet algorithme la fonction objectif est la fonction log-perte du modèle (mesure de la distance entre les valeurs observées et les valeurs calculées). Ajouter une pénalité signifie mutliplier le deuxième terme de la fonction objectif par cette pénalité pour ajuster le résultat. Un alpha élevé va réduire la variance qui est un signe d'overfitting. A l'inverse un alpha plus faible va réduire le biai, signe d'un underfitting. Nous choisissons trois valeurs pour alpha : 0.1, 0.01, 0.2, filtrées après plusieurs tentatives.
# 
# Le solver adam comporte 3 caractéristiques qui lui sont propres : beta_1, beta_2 et epsilon.
# Les betas sont compris entre 0 et 1 exclus. Ces paramètres permettent de faire varier le delais de la décroissance de moyennes mobiles utilisés par le solver, ils sont appelés « taux de décroissance exponentielle » et doivent faire tendre le biais des moments vers 0. Beta_1 permet de faire varier le premier moment et Beta_2 le second. Nous avons choisis de prendre deux paramètres pour ces betas. Le premier choix a été de prendre les deux betas proches de 1 (0.9 et 0.999) qui est la valeur de default de l'algorithme. Nous avons aussi chosis de selectionner des betas plus bas (0.5 et 0.555) pour voir si cela ne permettait pas d'avoir une convergence plus efficace et plus rapide du solver en cas d'un gradient faible.
# L'epsilon doit être très petit et prévient d'une division par 0 lors des calculs du solver.  Nous avons choisis de lui donner trois valeurs 1e-8, 1e-10, 1e-5. En effet, nous avons constaté dans différents tests que l’epsilon optimal devait se situer entre 1e-4 et 1e-10.
# 

# In[43]:


#All
param_solver = ['lbfgs','sgd','adam']
param_activation =[ 'relu','identity','logistic']
param_hiddenlayer = [10, 20, 50] #100 écarté
param_alpha = [0.1, 0.01, 0.2] #Pénalité L2 

#adam
param_beta1 = [ 0.5, 0.9]
param_beta2 = [ 0.555, 0.999]
param_epsilon = [1e-8, 1e-10, 1e-5]


# In[44]:


#KFOLDS pour la cross validation
kf = KFold(n_splits=4)

#DataFrame des métriques
metrics_list = pd.DataFrame()


# ### Fit du modèle
# Nous séparons les solvers dans des cellules séparées car ils n'ont pas les mêmes paramètres. Nous avons laissé en commentaire les lignes de code qui ont été éliminées au fur et à mesure de la calibratio. 

# #### Solver : lbfgs

# In[ ]:


t = datetime.now()

#Paramètres testés
for act in param_activation:
    print('activation = ' + act)
    for hl in param_hiddenlayer:
        print('hiddenlayer =' + str(hl))

        for al in param_alpha:
            print('alpha =' + str(al)+  str(datetime.now()-t))
            t = datetime.now()
            
            i=1
            #Crossval : 4 folds de la base train
            for train_index, valid_index in kf.split(X_train, y_train_norm_label):
                #BASES TRAIN ET VALID DE LA CROSS VALIDATION
                Xtrain, Xvalid = X_train.iloc[train_index], X_train.iloc[valid_index]
                #Base train et valid normalisées
                ytrain_norm, yvalid_norm = y_train_norm_label.iloc[train_index], y_train_norm_label.iloc[valid_index]
                #Bases train et valid non normalisées
                ytrain, yvalid = y_train_label.iloc[train_index], y_train_norm_label.iloc[valid_index]
                
                #PARAMETRAGE DU MLP
                mlp = MLPClassifier(hidden_layer_sizes= hl,max_iter=10000, solver= param_solver[0],
                                   activation= act, alpha= al) 
                
                #FIT DU MLP SUR DONNES NORMALISEES
                #print('Fit MLP NORM...')
                #mlp.fit(Xtrain, ytrain_norm.values.ravel()) 
                #predictions = mlp.predict(Xvalid)
                #metrics_list = metrics_list.append(model_metrics('MLP_Classifier', yvalid_norm, predictions, 
                                                                #'YES','NO', param_solver[0], i, act, hl, al))
                
                #FIT DU MLP SUR DONNES NON NORMALISEES 
                #print('Fit MLP NON NORM...')
                #mlp.fit(Xtrain, ytrain.values.ravel())
                #predictions = mlp.predict(Xvalid)
                #metrics_list = metrics_list.append(model_metrics('MLP_Classifier', yvalid, predictions, 
                                                                #'NO','NO', param_solver[0], i, act, hl, al))
                
                #SMOT : oversampling
                sm = SMOTE(random_state=2)
                Xtrain_smot_n, ytrain_norm_smot = sm.fit_sample(Xtrain, ytrain_norm.values.ravel()) #Normalized
                Xtrain_smot, ytrain_smot = sm.fit_sample(Xtrain, ytrain.values.ravel()) # Not normalized
                
                #FIT DU MLP SUR DONNES NORMALISEES ET SMOT
                print('Fit MLP SMOT  NORM...')
                mlp.fit(Xtrain_smot_n, ytrain_norm_smot) 
                predictions = mlp.predict(Xvalid)
                metrics_list = metrics_list.append(model_metrics('MLP_Classifier', yvalid_norm, predictions, 
                                                                'YES','YES', param_solver[0], i, act, hl, al))
                
                #FIT DU MLP SUR DONNES NON NORMALISEES ET SMOT
                #print('Fit MLP SMOT NON NORM...')
                #mlp.fit(Xtrain_smot, ytrain_smot)
                #predictions = mlp.predict(Xvalid)
                #metrics_list = metrics_list.append(model_metrics('MLP_Classifier', yvalid, predictions, 
                                                                #'NO','YES', param_solver[0], i, act, hl, al))
                
                
                print(' time elapsed :' +  str(datetime.now()-t))
                i+=1


# #### Solver : adam

# In[ ]:


t = datetime.now()

#Paramètres testés
for act in param_activation:
    print('activation = ' + act)
    
    for hl in param_hiddenlayer:
        print('hiddenlayer =' + str(hl))
        
        for al in param_alpha:
            print('alpha =' + str(al))

            for b1 in param_beta1:
                print('beta1 = ' + str(b1))
                
                for b2 in param_beta2:
                    print('beta2 =' + str(b2) )

                    for ep in param_epsilon:
                        print('epsilon =' + str(ep))

                        i=1
                        #CROSS VALIDATION
                        for train_index, valid_index in kf.split(X_train, y_train_norm_label):

                                Xtrain, Xvalid = X_train.iloc[train_index], X_train.iloc[valid_index]
                                #Base Train normalisée ou non 
                                ytrain_norm, yvalid_norm = y_train_norm_label.iloc[train_index],y_train_norm_label.iloc[valid_index]
                                ytrain, yvalid = y_train_label.iloc[train_index],y_train_label.iloc[valid_index]
                                
                                #Paramétrage du MLP
                                mlp = MLPClassifier(hidden_layer_sizes= hl,max_iter=10000, solver= param_solver[2],
                                                   activation= act, alpha= al,epsilon= ep, beta_1= b1, beta_2= b2) 
                                
                                #FIT DU MLP SUR DONNEES NORMALISEES
                                #print('Fit MLP NORM...')
                                #mlp.fit(Xtrain, ytrain_norm.values.ravel()) 
                                #predictions = mlp.predict(Xvalid)
                                #metrics_list = metrics_list.append(model_metrics('MLP_Classifier', yvalid_norm, predictions, 
                                                                            #'YES', 'NO',param_solver[2], i, act, hl, al, b1 = b1,
                                                                                #b2 = b2, ep= ep))
                                
                                #FIT DU MLP SUR DONNEES NON NORMALISEES
                                #print('Fit MLP NON NORM...')
                                #mlp.fit(Xtrain, ytrain.values.ravel()) 
                                #predictions = mlp.predict(Xvalid)
                                #metrics_list = metrics_list.append(model_metrics('MLP_Classifier', yvalid, predictions, 
                                                                            #'NO','NO', param_solver[2], i, act, hl, al, b1 = b1,
                                                                             #   b2 = b2, ep= ep))
                            
                                
                                
                                #SMOT : oversampling
                                sm = SMOTE(random_state=2)
                                Xtrain_smot_n, ytrain_norm_smot = sm.fit_sample(Xtrain, ytrain_norm.values.ravel()) #Normalized
                                Xtrain_smot, ytrain_smot = sm.fit_sample(Xtrain, ytrain) # Not normalized

                                #FIT DU MLP SUR DONNEES NORMALISEES ET SMOT
                                print('Fit MLP SMOT NORM...')
                                mlp.fit(Xtrain_smot_n, ytrain_norm_smot)
                                predictions = mlp.predict(Xvalid)
                                metrics_list = metrics_list.append(model_metrics('MLP_Classifier', yvalid_norm, predictions, 
                                                                            'YES', 'YES', param_solver[2], i, act, hl, al, b1 = b1,
                                                                                b2 = b2, ep= ep))
                                
                                #FIT DU MLP SUR DONNEES NON NORMALISEES ET SMOT
                                #print('Fit MLP SMOT NON NORM...')
                                #mlp.fit(Xtrain_smot, ytrain_smot) 
                                #predictions = mlp.predict(Xvalid)
                                #metrics_list = metrics_list.append(model_metrics('MLP_Classifier', yvalid, predictions, 
                                                                            #'NO','YES', param_solver[2], i, act, hl, al, b1 = b1,
                                                                               # b2 = b2, ep= ep))
                                
                                print('Time elapsed :' +  str(datetime.now()-t))
                                t = datetime.now()
                                i+=1 #Numéro du fold


# In[ ]:


metrics_list.head()


# In[ ]:


metrics_list.to_csv('metrics.csv')


# #### Récupération des métriques en csv
# Nous avons sauvegardés les métriques en csv afin de les récupérer plus tard sans relancer les fits.

# In[45]:


metrics_list = pd.read_csv('metrics.csv').drop('Unnamed: 0',axis = 1)
metrics_list.head()


# #### Moyennisation des folds  et classement des modèles
# Nous rassemblons les paramètres afin de les grouper pour moyenner les métriques sur les 4 folds. 
# Pour un 8-uplet de paramètres il y a 4 fits pour les 4 folds. Nous moyennons les 3 métriques sur ces 4 fits .

# In[46]:


metrics_list['allparam'] = metrics_list[['1_normalisation', '1_SMOT','2_solver',
                                          '3_activation','3_hiddenlayer','3_alpha',
                                          '4_beta1ADAM','4_epsilonADAM']].astype(str).agg('_'.join, axis=1)
model_ranking = metrics_list.groupby('allparam').mean().sort_values(by ='0_precision', ascending = False)


# Nous avons au total 216 n-uplets différents triés dans l'ordre du meilleur critère AUC. Nous décidons de maximiser les critères AUC et précision pour choisir notre meilleur modèle. 
# 
# 
# ### Choix des critères
# 
# #### Critère AUC
# 
# Le critère AUC, qui correspond à l'aire sous la courbe ROC, nous permet d'avoir une mesure aggrégée des performances du modèle. La courbe ROC (receiver operating characteristic) représente les performances de nos modèles de classification pour tous les seuils de classification (seuil auquel l'algorithme va définir un rendement comme 1 ou 0). Les performances sont représentées comme le taux de vrais positifs en fonction du taux de faux positifs pour différents seuils. <br>
# 
# Taux de vrais positifs :  $TVP= \frac{VP}{VP+FN}$ <br>
# 
# Taux de faux positifs : $TFP = \frac{FP}{FP+VN}$ <br>
# 
# Le critère AUC est donc l'intégrale sous la courbe ROC. Cette mesure est donc indépendante du seuil de classification. 
# 
# 
# #### Critère précision
# La précision est le nombre de vrais positifs sur le nombre de classés positifs : $\frac{VP}{VP+FP}$ <br>
# Elle permet de juger de l'exactitude du modèle. Maximiser cette métrique permet de minimiser la part de faux positifs.
# 
# 
# #### Critère rappel
# Le rappel est le nombre de vrais positifs sur le nombre total de positifs : $\frac{VP}{VP+FN}$ <br>
# Il permet de mesurer l'exhaustivité des prédictions, de savoir si tous les positifs ont bien été classés positifs.
# 
# 
# #### Choix :
# Dans notre cas, nous voulons un algorithme permettant de lancer un ordre de trade long lorsque le rendement est supérieur à 2%. Nous appliquons donc le principe de prudence en choisissant de minimiser la part de faux positifs ce qui réduirait l'achat d'actions à faible rendement. C'est pourquoi nous choisissons le critère précision comme référence. Notre deuxième critère de référence est l'AUC car il permet de prendre en compte des performances agrégées. De plus, l'information du critère rappel est contenue dans le critère AUC puisque le rappel est également le taux de vrais positifs.

# ## Résultat
# Une fois les métriques choisies, nous pouvons nous concentrer sur le classement établi plus haut.

# In[47]:


#AUC Maximal
model_ranking[model_ranking['0_precision']==model_ranking['0_precision'].max()]


# In[48]:


model_ranking.head(10)


#  Le modèle arrivant en tête pour le critère Précision utilise le solver lbfgs et des données non over-sampleés. Cependant, en regardant plus loin dans le classement nous voyons qu'il semble être une exception puisque tous les autres modèles du top 10 utilisent le solver adam. 
# Nous éliminons donc le solver lbfgs de notre décision finale. Nous nous replions sur le modèle arrivant en deuxième position qui montre en plus un critère AUC plus élevé que le premier. 

# In[49]:


#Modèle retenu
model_retained = model_ranking.reset_index().iloc[1,:]
model_retained


# Le modèle retenu utilise finalement le solver Adam, sur des données normalisées et oversamplées, une fonction d'activation logistique, 20 neurones dans une couche cachée, une pénalité $\alpha$ égale à 0.2, des $\beta$ égaux à 0.5 et 0.777 et un $\varepsilon$ égal à 1e-10.

# In[99]:


#Oversampling
sm = SMOTE(random_state=2)
X_final, y_final = sm.fit_sample(X_train, y_train_norm_label.values.ravel()) 

#Paramétrage
solv = 'adam'
act= 'logistic'
hl = model_retained['3_hiddenlayer'].astype('int32')
al = model_retained['3_alpha']
ep = model_retained['4_epsilonADAM']
b1 = model_retained['4_beta1ADAM']
b2 = model_retained['4_beta2ADAM']


mlp = MLPClassifier(hidden_layer_sizes= hl,max_iter=10000, solver= solv,
                                                   activation= act, alpha= al,epsilon= ep, beta_1= b1, beta_2= b2) 

#Fit
mlp.fit(X_final, y_final)

                                


# In[100]:


#Prédictions sur la base TEST 
predictions = mlp.predict(X_test)
metrics_retained = pd.DataFrame(model_metrics('ALL_retained', y_test_label, predictions,'YES', 'YES', param_solver[2], i, act, hl, al, b1 = b1,
              b2 = b2, ep= ep))
metrics_retained


# ## Importance des variables
# Nous testons maintenant l'importance des variables dans le modèle après avoir choisi les meilleurs paramètres. 

# In[101]:


#on remplace volontairement cette n-ième variable par un tirage aléatoire avec remise sur cette variable : 
from random import choices

words_importance = metrics_retained.copy()
nb_rows = X_test.shape[0]

for n in range(X_test.shape[1]):
    X_tmp = X_test.copy()
    X_tmp.iloc[:,n] = choices(list(X_tmp.iloc[:,n].unique()), k= nb_rows)
    pred = mlp.predict(X_tmp)
    words_importance = words_importance.append(model_metrics(X_test.columns[n], y_test_label, pred ,'YES', 'YES', param_solver[2],
                                                           i, act, hl, al, b1 = b1,b2 = b2, ep= ep))

words_importance.sort_values(by='0_auc', ascending=False)


# Aucun modèle avec variable écartée n'a un meilleur score AUC que le modèle avec toutes les variables. Nous ne retirons donc aucune variable
