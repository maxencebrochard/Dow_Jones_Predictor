
# coding: utf-8

# Bienvenue sur mon algorithme de prédicteur de tendance boursière.
# 
# Le but de ce projet est de prédire la hausse ou la baisse du cours du Dow Jones à l'aide d'informations agrégées par le site Reddit.com. Pour cela nous disposons d'un data set disponible sur https://www.kaggle.com/aaron7sun/stocknews. Nous ferons dans un premier temps une analyse descriptive du jeu de données avant de passer à la partie prédictive du projet.
# 
# Le site Reddit.com est un site web communautaire de partage de signets permettant aux utilisateurs de soumettre leurs liens et de voter pour les liens proposés par les autres utilisateurs (source Wikipedia). Les informations comprises dans le data set proviennent de la page Reddit World News (https://www.reddit.com/r/worldnews/?hl=). Par jour d'activité boursière sur le Dow Jones, le data set comprend les 25 'Top' news déterminées par le upvote ou downvote des utilisateurs de Reddit. Ces 25 news correspondent à 25 des 26 variables présentes dans le jeu de données.
# 
# Concernant notre indice de référence à prédire, le Dow Jones Industrial Average est un indice boursier américain qui reflète la valorisation financière des 30 sociétés les plus importantes en matière de capitalisation boursière. La 26ème variable du jeu de données correspond à une indicatrice qui détermine si le Dow Jones s'est maintenu à l'équilibre ou a augmenté (Label=1), ou si le Dow Jones a baissé (Label=0)
# 
# Avant de commencer l'analyse du jeu de données, observons la courbe de valorisation du Dow Jones qui correspond aux dates de notre data set. En effet, notre data set s'étendant de 2008 à nos jours, il semble opportun d'observer la courbe du DJ sur cette période. Ci-dessous, nous obtenons un aperçu de l'évolution du cours du Dow Jones grâce à l'api Quandl qui permet de télécharger les cotations de produits financiers.

# In[4]:

import quandl #pip install quandl ou conda install -c anaconda quandl=3.0.1
import matplotlib.pyplot as plt
import numpy as np

data = quandl.get("BCB/UDJIAD1", authtoken="Fyqdkmx8zza_-DJLxCNs" , trim_start = "2008-01-01", trim_end = "2016-10-01", collapse = "daily")
plt.plot(data)
plt.title('Dow Jones Jones Industrial Average - Cotation de 2008 à nos jours', y=1.08)
plt.show()


# Nous pouvons observer que le jeu de données commencent dès la crise boursière de 2008 et l'effondrement des cotations qui s'en suivit. Dès lors, nous sommes en mesure de nous demander si un modèle qui est entrainé sur une période de fortes variations boursières sera à même de correctement prédire. En effet, on peut s'attendre à ce que le lexique utilisé dans les informations financières de cette période sera le même que celui utilisé en sortie de crise.

# Les commandes ci-dessous permettent d'importer le data set et d'afficher un aperçu de celui-ci.
# Le data set contient : 
# - la date (un jour par ligne). Le data set s'étend du 8 août 2008 au 1er juillet 2016
# - les 25 news quotidiennes les plus importantes d'après le site reddit (Top 1 à Top 25)
# - la hausse ou la baisse du Dow Jones matérialisé pa une indicatrice (Label)

# In[9]:

import pandas
import os.path as ospath
import os
os.getcwd()
#assert ospath.isfile("./DataSet/Combined_News_DJIA.csv") , "Combined_News_DJIA not found."
#df = pandas.read_csv("./DataSet/Combined_News_DJIA.csv", sep=",")
#df = pandas.read_csv("C:/Users/Maxence/Desktop/Projets/nlptrd/DataSet/Combined_News_DJIA.csv", sep=",")


# In[3]:

import pandas

df = pandas.read_csv("C:/Users/Maxence/Desktop/Projets/nlptrd/DataSet/Combined_News_DJIA.csv", sep=",")
row,_ = df.shape
df.head()


# 
# 

# In[4]:

#On transforme toutes les news du data set en une seule chaîne de caractères. L'instruction try except permet à l'algorithme 
#de continuer même en cas de valeurs manquantes.
content = ""
for i in range(row):
    try:
        content = content + " ".join(word for word in df.ix[i, 2:27])
    except:
        pass
content[:500] #On affiche les 500 premiers caractères du data set concaténé 


# Maintenant que nous avons condensé toute l'information, nous allons explorer le jeu de données

# In[6]:

print("Le data set contient ", df.shape[0], " lignes et ", df.shape[1], " colonnes.")
print("L'information disponible contient :", len(content), " mots.")
print("Le data set s'étend du :", min(df.ix[:,0]), "au ", max(df.ix[:,0]))


# Nous effectuons une opération de tokenize qui nous permet de parser la chaine de caratères et de conserver uniquement les mots
# (on y supprime ici les chiffres).

# In[8]:

from nltk.tokenize import RegexpTokenizer
#le tokenizer utilise des expressions régulières et ne conserve que les mots
tokenizer = RegexpTokenizer("[a-zA-Z'`éèî]+")

#On garde les mots en minuscules et en ayant supprimer les espaces
tokens = tokenizer.tokenize(content.lower().strip())
print("Il reste ", len(tokens), " après tokenization")


# In[9]:

tokens[:500]


# In[ ]:

words_count = sorted(words.items(), key=operator.itemgetter(1), reverse = True)
tokens = [word for word in tokens if word not in exclu]
len(tokens)

removed_words = [word for word in tokens if word in exclu]


# In[10]:

import spacy

#spaCy est une librairie donnant accès à de nombreux outils de NLP 
nlp  = spacy.load('en')
doc = nlp(content)

#spaCy tokenize le contenu du data set et reconnaît le type de mot en question (Date, Personne, Chiffre...)


# In[ ]:

for num, entity in enumerate(doc.ents):
    print ('Entity {}:'.format(num + 1), entity, '-', entity.label_)
    print ('')


# In[7]:

from nltk.tokenize import RegexpTokenizer

#Create a tokenizer that can split strings and remove unwanted symbols
tokenizer = RegexpTokenizer("[a-zA-Z'`éèî]+")
tokens = []
words = {}


# In[12]:

#Nous avons déterminé au préablable une liste de stop words pour garder le contrôle des mots qui ne seront pas inclus 
#dans l'analyse.

exclu = {'to', 'in', 'and', 'for', 'is', 'by', 'that', "'", 'has', 'with', 'from', 
'as', 'are', 'at', 'have', 'be', 'it', 'after', 'us', 'an', 'b', 'over', 'says', 
'was', 'will', 'not', 'its', 'who', 'their', 's', 'they', 'u', 'been', 'his', 
'more', 'up', 'against', 'than', 'out', 'war', 'into', 'the', 'to', 'of', 'in', 'a',
'and', 'for', 'in', 'is', 'by', 'that', 'with', 'from', "'", 'has', 'as', 'are',
'be', 'it', 'an', 'this', 'he', 'about', 'were', 'on', 'but', 'or', 'amp', 'you',
'them', 'i', "b'", 'so', 'do', 'our'}


# In[13]:

#Tokenize the data set and creates a dictionnary (words{}) to get the most used words
for i in range(row):
    for j in range(1,26):
        try:
            for word in tokenizer.tokenize(df.ix[i, "Top"+str(j)]):
                word = word.lower().strip()
                if word not in exclu:
                    if word in words:
                        words[word] += 1
                    else:
                        words[word] = 1
        except:
            print('########### error on line ' + str(i) + 'and column ' + str(j) +' ###########') 
            #Errors are due to missing values


# In[17]:

import operator
#Sort the words by number of occurences
words_count = sorted(words.items(), key=operator.itemgetter(1), reverse = True)
words_count


# In[28]:

#Creates a list of variables corresponding to the most used words. We will use those variables in our restructured data set
num_var = 200
words_list = [w for w,n in words_count[:num_var]]


# In[ ]:




# In[ ]:



