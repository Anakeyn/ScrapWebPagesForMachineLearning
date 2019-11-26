# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:39:18 2019

@author: Pierre
"""
##########################################################################
# ScrapWebPagesForMachineLearning
# Auteur : Pierre Rouarch - Licence GPL 3
# Scraping de page Web pour enrichir le Machine Learning
# sur un univers de concurrence 
# Dans cette partie on va créer de nouvelles variables explicatives 
# en récupérant le contenu des pages.
###################################################################
# On démarre ici 
###################################################################
#Chargement des bibliothèques générales utiles
import numpy as np #pour les vecteurs et tableaux notamment
import pandas as pd  #pour les Dataframes ou tableaux de données
import os

from urllib.parse import urlparse #pour parser les urls
import nltk # Pour le text mining

import requests #idem pour requete de la page  finalement on choisit celui-ci
import time  #pour calculer le 'temps' de chargement de la page
from bs4 import BeautifulSoup, SoupStrainer #pour le traitement des balises.
from bs4.element import Comment #pour récupérer les commentaires du source html
import re  #pour les expressions regulieres
#pip install unicodedata
import unicodedata  #pour décoder les accents 
import gc #pour vider la memoire


print(os.getcwd())  #verif
#mon répertoire sur ma machine - nécessaire quand on fait tourner le programme 
#par morceaux dans Spyder.
#myPath = "C:/Users/Pierre/MyPath"
#os.chdir(myPath) #modification du path
#print(os.getcwd()) #verif



############################################
# Calcul de la somme des tf*idf
#somme des TF*IDF pour chaque colonne de tokens calculée avec TfidfVectorizer
def getSumTFIDFfromDFColumn(myDFColumn) :
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = myDFColumn.apply(' '.join)
    vectorizer = TfidfVectorizer(norm=None)
    X = vectorizer.fit_transform(corpus)
    return np.sum(X.toarray(), axis=1)





#########################################################################
#  Enrichissement des données avec le contenu des pages.
###########################################################
# on va scraper les pages que l'on a avec beautifoulsoup4 et 
# requests :
# http://docs.python-requests.org/en/master/
###############################################################
    
#Lecture du fichier précédent  ############
dfQPPS1 = pd.read_json("dfQPPS1-MAI.json")
dfQPPS1.query
dfQPPS1.info() # 14315 enregistrements.

#on filtre les extensions non html
extensionsToCheck = ('.7z','.aac','.au','.avi','.bmp','.bzip','.css','.doc',
                     '.docx','.flv','.gif','.gz','.gzip','.ico','.jpg','.jpeg',
                     '.js','.mov','.mp3','.mp4','.mpeg','.mpg','.odb','.odf',
                     '.odg','.odp','.ods','.odt','.pdf','.png','.ppt','.pptx',
                     '.psd','.rar','.swf','.tar','.tgz','.txt','.wav','.wmv',
                     '.xls','.xlsx','.xml','.z','.zip')

indexGoodFile=dfQPPS1['page'].apply(lambda x : not x.endswith(extensionsToCheck) )
dfQPPS2=dfQPPS1.iloc[indexGoodFile.values]
dfQPPS2.reset_index(inplace=True, drop=True)
dfQPPS2.info() #14162 un peu moins que précédement - 150 environ

#######################################################
# On ne va scraper les pages qu'une fois  !
########################################################
myPagesToScrap = dfQPPS2['page'].unique()
dfPagesToScrap= pd.DataFrame(myPagesToScrap, columns=["page"])
dfPagesToScrap.size #9709 pages




#######################################################
########## Lecture des pages !!!!  peut être long !!! 


#ajout de nouvelles variables
dfPagesToScrap['statusCode'] = ''
dfPagesToScrap['html'] = ''  #servira par la suite.
dfPagesToScrap['encoding'] = ''  #servira par la suite.
dfPagesToScrap['elapsedTime'] = np.nan

for i in range(0,len(dfPagesToScrap)) :
    url = dfPagesToScrap.loc[i, 'page']
    print("Page i = "+url+" "+str(i))
    startTime = time.time()
    try:
        #html = urllib.request.urlopen(url).read()$
        r = requests.get(url, timeout=5)  #finalement on prend request
        dfPagesToScrap.loc[i,'statusCode'] = r.status_code
        print('Status_code '+str(dfPagesToScrap.loc[i,'statusCode']))
        if r.status_code == 200 :
            dfPagesToScrap.loc[i,'encoding'] = r.encoding #peut servir 
            dfPagesToScrap.loc[i, 'html'] = r.text  #on conserve tout le contenu 
            #au format texte r.text - pas bytes : r.content
            print("ok page ") 
    except:
        print("Erreur page ") 
    endTime= time.time()   
    dfPagesToScrap.loc[i, 'elapsedTime'] =  endTime - startTime
    
           
      
#SVG dfPagesToScrap        
   #Sauvegarde 
dfPagesToScrap.to_json("dfPagesToScrap.json")  


#Relecture ############
dfPagesToScrap = pd.read_json("dfPagesToScrap.json")
dfPagesToScrap.query
dfPagesToScrap.info() # env 9709  enregistrements.     
        
        
#ici faire le merge avec   dfQPPS2    ->  dfQPPS3   
dfQPPS3 = pd.merge(dfQPPS2, dfPagesToScrap, on='page', how='left')    
#on ne garde que les status code = 200   
dfQPPS3 = dfQPPS3.loc[dfQPPS3['statusCode'] == 200]   
dfQPPS3.reset_index(inplace=True, drop=True)    
dfQPPS3.info() # 12207 enregistrements
dfQPPS3 = dfQPPS3.dropna()  #on vire les lignes qui comportent au moins un na
dfQPPS3.reset_index(inplace=True, drop=True) 
dfQPPS3.info() #12194 enregistrements
#Sauvegarde 
dfQPPS3.to_json("dfQPPS3.json")  
##########################################

###########################################################################
# Enrichissement avec des variables créées à partir du contenu des pages
###########################################################################

#Relecture ############
dfQPPS3 = pd.read_json("dfQPPS3.json")
dfQPPS3.query
dfQPPS3.info() # 12194 enregistrements
      
#vider la memoire
dir()
del dfPagesToScrap
del dfQPPS1
del dfQPPS2

gc.collect()

     
#####Récupération de données de balises
def getStringfromTag(tag="h1") :
    theTag = soup.find_all(tag)
    myTag = ""
    for x in theTag:
        myTag= myTag + " " + x.text.strip()
    return myTag.strip()   

#pour enlever les éléments non visibles et les commentaires 
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

 

def strip_accents(text, encoding='utf-8'):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode(encoding)
    return str(text)

tokenizer = nltk.RegexpTokenizer(r'\w+')  #définition du tokeniser pour séparation des mots

###############################################################
# Intialisation et définition de nouvelles variables
###############################################################
dfQPPS3['lenTokensQuery'] = 0.0  #nombre de mots dans la requete 
#pour le titre
dfQPPS3['Title'] = ""  #Titre String vide
dfQPPS3['lenTitle'] = 0.0  #taille en caractères du titre
dfQPPS3.apply(lambda x: [], axis=1) #liste de mots du titre
dfQPPS3['lenTokensTitle'] = 0.0  #nombre de mots du titre
dfQPPS3['lenTokensQueryInTitleFrequency'] = 0.0 #fréquence de mots de la requete dans le titre
#sumTFIDF
dfQPPS3['sumTFIDFTitle'] = 0.0   


#pour la description
dfQPPS3['Description'] = ""  #String vide
dfQPPS3['lenDescription'] = 0.0  #taille en caractères
dfQPPS3['tokensDescription'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensDescription'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInDescriptionFrequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFDescription'] = 0.0   
#pour H1
dfQPPS3['H1'] = ""  #String vide
dfQPPS3['lenH1'] = 0.0  #taille en caractères
dfQPPS3['tokensH1'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensH1'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInH1Frequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFH1'] = 0.0  
#pour H2
dfQPPS3['H2'] = ""  #String vide
dfQPPS3['lenH2'] = 0.0  #taille en caractères
dfQPPS3['tokensH2'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensH2'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInH2Frequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFH2'] = 0.0  
#pour H3
dfQPPS3['H3'] = ""  #String vide
dfQPPS3['lenH3'] = 0.0  #taille en caractères
dfQPPS3['tokensH3'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensH3'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInH3Frequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFH3'] = 0.0  
#pour H4
dfQPPS3['H4'] = ""  #String vide
dfQPPS3['lenH4'] = 0.0  #taille en caractères
dfQPPS3['tokensH4'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensH4'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInH4Frequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFH4'] = 0.0  
#pour H5
dfQPPS3['H5'] = ""  #String vide
dfQPPS3['lenH5'] = 0.0  #taille en caractères
dfQPPS3['tokensH5'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensH5'] = 0.0  #nombre de mots en H5
dfQPPS3['lenTokensQueryInH5Frequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFH5'] = 0.0  
#pour H6
dfQPPS3['H6'] = ""  #String vide
dfQPPS3['lenH6'] = 0.0  #taille en caractères
dfQPPS3['tokensH6'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensH6'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInH6Frequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFH6'] = 0.0  
#pour B
dfQPPS3['B'] = ""  #String vide
dfQPPS3['lenB'] = 0.0  #taille en caractères
dfQPPS3['tokensB'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensB'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInBFrequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFB'] = 0.0  
#pour I
dfQPPS3['I'] = ""  #String vide
dfQPPS3['lenI'] = 0.0  #taille en caractères
dfQPPS3['tokensI'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensI'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInIFrequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFI'] = 0.0  
#pour EM
dfQPPS3['EM'] = ""  #String vide
dfQPPS3['lenEM'] = 0.0  #taille en caractères
dfQPPS3['tokensEM'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensEM'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInEMFrequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFEM'] = 0.0  
#pour Strong
dfQPPS3['Strong'] = ""  #String vide
dfQPPS3['lenStrong'] = 0.0  #taille en caractères
dfQPPS3['tokensStrong'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensStrong'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInStrongFrequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFStrong'] = 0.0  
#pour Body
dfQPPS3['Body'] = ""  #String vide
dfQPPS3['lenBody'] = 0.0  #taille en caractères
dfQPPS3['tokensBody'] = dfQPPS3.apply(lambda x: [], axis=1)  #liste de mots
dfQPPS3['lenTokensBody'] = 0.0  #nombre de mots 
dfQPPS3['lenTokensQueryInBodyFrequency'] = 0.0 #fréquence de mots de la requete 
#sumTFIDF 
dfQPPS3['sumTFIDFBody'] = 0.0  

#Liens internes et externes
dfQPPS3['nbrInternalLinks'] = 0
dfQPPS3['nbrExternalLinks'] = 0



#i=0 #pour test
#len(dfQPPS3)  12194
#on boucle pour décoder le code HTML des pages web
for i in range(0,len(dfQPPS3)) :
    print("Page query i = "+dfQPPS3.loc[i, 'page']+" "+dfQPPS3.loc[i, 'query']+" "+str(i))
    #autre méthode brute pour récupérer l'encoding
    #encoding = re.findall(r'<meta.*?charset=["\']*(.+?)["\'>]', dfQPPS3['html'][i], flags=re.I)[0]
    encoding = dfQPPS3.loc[i, 'encoding'] #on l'avait conservé.
    if (type(encoding) == float and np.isnan(encoding)) or encoding=='binary' : #ça arrive !
        encoding="utf-8"
    #Récuperation de la recquete
    queryNoAccent= strip_accents( dfQPPS3.loc[i,'query'], encoding).lower()
    tokensQueryNoAccent = tokenizer.tokenize(queryNoAccent)
    dfQPPS3.loc[i, 'lenTokensQuery']=len(tokensQueryNoAccent) #taille en mots
    try:
        soup = BeautifulSoup(dfQPPS3.loc[i, 'html'], 'html.parser')
    except :
        soup="" 
    if len(soup) != 0 :  #si on a de la  soup :-) !!!

        #Titre 
        try: 
            myTitle = strip_accents(soup.title.string, encoding).lower()
        except:
            myTitle=""
            
        print("Title="+myTitle) 
        dfQPPS3.loc[i, 'Title'] = myTitle   
        dfQPPS3.loc[i, 'lenTitle'] = len(myTitle)
        tokensTitle= tokenizer.tokenize(myTitle)
        #taille en mots
        dfQPPS3.loc[i,'lenTokensTitle'] = len(tokensTitle)
        #nombre de mots de la query dans le titre
        lenTokensQueryInTitle = len([word for word in tokensTitle if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensTitle'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInTitleFrequency'] = lenTokensQueryInTitle/dfQPPS3.loc[i,'lenTokensTitle']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInTitleFrequency'] = 0
        
        #Description
        myDescription=""
        if (len(soup.select('meta[name="description"]')) >0) :
            try:
                myDescription = soup.select('meta[name="description"]')[0].attrs['content']
            except:
              myDescription=""
        else :
            myDescription=""
        print("Description="+myDescription) 
        myDescription = strip_accents(myDescription, encoding).lower()
        dfQPPS3.loc[i, 'lenDescription'] = len(myDescription)
        tokensDescription= tokenizer.tokenize(myDescription)
        dfQPPS3.loc[i, 'Description'] = myDescription
        #taille en mots
        dfQPPS3.loc[i,'lenTokensDescription'] = len(tokensDescription)
        lenTokensQueryInDescription = len([word for word in tokensDescription if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensDescription'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInDescriptionFrequency'] = lenTokensQueryInDescription/dfQPPS3.loc[i,'lenTokensDescription']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInDescriptionFrequency'] = 0       
        #H1
        myH1 =  getStringfromTag("h1")
        myH1 = strip_accents(myH1, encoding).lower()
        dfQPPS3.loc[i, 'lenH1'] = len(myH1)
        tokensH1= tokenizer.tokenize(myH1)
        dfQPPS3.loc[i, 'H1'] = myH1
        #taille en mots
        dfQPPS3.loc[i,'lenTokensH1'] = len(tokensH1)
        lenTokensQueryInH1 = len([word for word in tokensH1 if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensH1'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInH1Frequency'] = lenTokensQueryInH1/dfQPPS3.loc[i,'lenTokensH1']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInH1Frequency'] = 0
        #H2
        myH2 =  getStringfromTag("h2")
        myH2 = strip_accents(myH2, encoding).lower()
        dfQPPS3.loc[i, 'lenH2'] = len(myH2)
        tokensH2= tokenizer.tokenize(myH2)
        dfQPPS3.loc[i, 'H2'] = myH2
        #taille en mots
        dfQPPS3.loc[i,'lenTokensH2'] = len(tokensH2)
        lenTokensQueryInH2 = len([word for word in tokensH2 if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensH2'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInH2Frequency'] = lenTokensQueryInH2/dfQPPS3.loc[i,'lenTokensH2']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInH2Frequency'] = 0
        #H3
        myH3 =  getStringfromTag("h3")
        myH3 = strip_accents(myH3, encoding).lower()
        dfQPPS3.loc[i, 'lenH3'] = len(myH3)
        tokensH3= tokenizer.tokenize(myH3)
        dfQPPS3.loc[i, 'H3'] = myH3
        #taille en mots
        dfQPPS3.loc[i,'lenTokensH3'] = len(tokensH3)
        lenTokensQueryInH3 = len([word for word in tokensH3 if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensH3'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInH3Frequency'] = lenTokensQueryInH3/dfQPPS3.loc[i,'lenTokensH3']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInH3Frequency'] = 0
        #H4
        myH4 =  getStringfromTag("h4")
        myH4 = strip_accents(myH4, encoding).lower()
        dfQPPS3.loc[i, 'lenH4'] = len(myH4)
        tokensH4= tokenizer.tokenize(myH4)
        dfQPPS3.loc[i, 'H4'] = myH4
        #taille en mots
        dfQPPS3.loc[i,'lenTokensH4'] = len(tokensH4)
        lenTokensQueryInH4 = len([word for word in tokensH4 if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensH4'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInH4Frequency'] = lenTokensQueryInH4/dfQPPS3.loc[i,'lenTokensH4']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInH4Frequency'] = 0
        #H5
        myH5 =  getStringfromTag("h5")
        myH5 = strip_accents(myH5, encoding).lower()
        dfQPPS3.loc[i, 'lenH5'] = len(myH5)
        tokensH5= tokenizer.tokenize(myH5)
        dfQPPS3.loc[i, 'H5'] = myH5
        #taille en mots
        dfQPPS3.loc[i,'lenTokensH5'] = len(tokensH5)
        lenTokensQueryInH5 = len([word for word in tokensH5 if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensH5'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInH5Frequency'] = lenTokensQueryInH5/dfQPPS3.loc[i,'lenTokensH5']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInH5Frequency'] = 0
        #H6
        myH6 =  getStringfromTag("h6")
        myH6 = strip_accents(myH6, encoding).lower()
        dfQPPS3.loc[i, 'lenH6'] = len(myH6)
        tokensH6= tokenizer.tokenize(myH6)     
        dfQPPS3.loc[i, 'H6'] = myH6
        #taille en mots
        dfQPPS3.loc[i,'lenTokensH6'] = len(tokensH6)
        lenTokensQueryInH6 = len([word for word in tokensH6 if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensH6'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInH6Frequency'] = lenTokensQueryInH6/dfQPPS3.loc[i,'lenTokensH6']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInH6Frequency'] = 0
        #B
        myB =  getStringfromTag("b")
        myB = strip_accents(myB, encoding).lower()
        dfQPPS3.loc[i, 'lenB'] = len(myB)
        tokensB= tokenizer.tokenize(myB)
        dfQPPS3.loc[i, 'B'] = myB
        #taille en mots
        dfQPPS3.loc[i,'lenTokensB'] = len(tokensB)
        lenTokensQueryInB = len([word for word in tokensB if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensB'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInBFrequency'] = lenTokensQueryInB/dfQPPS3.loc[i,'lenTokensB']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInBFrequency'] = 0
        #I
        myI =  getStringfromTag("I")
        myI = strip_accents(myI, encoding).lower()
        dfQPPS3.loc[i, 'lenI'] = len(myI)
        tokensI = tokenizer.tokenize(myI)
        dfQPPS3.loc[i, 'I'] = myI
        #taille en mots
        dfQPPS3.loc[i,'lenTokensI'] = len(tokensI)
        lenTokensQueryInI = len([word for word in tokensI if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensI'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInIFrequency'] = lenTokensQueryInI/dfQPPS3.loc[i,'lenTokensI']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInIFrequency'] = 0
        #EM
        myEM =  getStringfromTag("em")
        myEM = strip_accents(myEM, encoding).lower()
        dfQPPS3.loc[i, 'lenEM'] = len(myEM)
        tokensEM= tokenizer.tokenize(myEM)
        dfQPPS3.loc[i, 'EM'] = myEM
        #taille en mots
        dfQPPS3.loc[i,'lenTokensEM'] = len(tokensEM)
        lenTokensQueryInEM = len([word for word in tokensEM if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensEM'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInEMFrequency'] = lenTokensQueryInEM/dfQPPS3.loc[i,'lenTokensEM']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInEMFrequency'] = 0
        #Strong
        myStrong =  getStringfromTag("strong")
        myStrong = strip_accents(myStrong, encoding).lower()
        dfQPPS3.loc[i, 'lenStrong'] = len(myStrong)
        tokensStrong= tokenizer.tokenize(myStrong)
        dfQPPS3.loc[i, 'Strong'] = myStrong
        #taille en mots
        dfQPPS3.loc[i,'lenTokensStrong'] = len(tokensStrong)
        lenTokensQueryInStrong = len([word for word in tokensStrong if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensStrong'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInStrongFrequency'] = lenTokensQueryInStrong/dfQPPS3.loc[i,'lenTokensStrong']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInStrongFrequency'] = 0
        #Tout le contenu de body
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)  
        myBody = " ".join(t.strip() for t in visible_texts)
        myBody=myBody.strip()
        myBody=re.sub(' +', ' ',myBody)
        myBody = strip_accents(myBody, encoding).lower()
        dfQPPS3.loc[i, 'lenBody'] = len(myBody)
        tokensBody= tokenizer.tokenize(myBody)
        dfQPPS3.loc[i, 'Body'] = myBody
        #taille en mots
        dfQPPS3.loc[i,'lenTokensBody'] = len(tokensBody)
        lenTokensQueryInBody = len([word for word in tokensBody if word in tokensQueryNoAccent])
        if dfQPPS3.loc[i,'lenTokensBody'] > 0 :
            dfQPPS3.loc[i, 'lenTokensQueryInBodyFrequency'] = lenTokensQueryInBody/dfQPPS3.loc[i,'lenTokensBody']
        else :
            dfQPPS3.loc[i, 'lenTokensQueryInBodyFrequency'] = 0
        #recupération des liens
        soupLinks = BeautifulSoup(dfQPPS3.loc[i, 'html'], 'html.parser', parse_only=SoupStrainer('a'))
        theLinks = [link['href'] for link in soupLinks if link.get('href')]
        myDomain=dfQPPS3.loc[i, 'uriNetLoc']
        nbrInternalLinks = 0
        nbrExternalLinks = 0
        for link  in theLinks :
            if len(link) > 0 :    #on ne prend que s'il y a qq chose dans le lien
                myLink = urlparse(link)        
                #print("Lien repéré = "+link)
                #print("Domaine repéré = "+myLink.netloc)
                if (myLink.netloc == myDomain or myLink.netloc == ""):
                    nbrInternalLinks += 1
                else:
                    nbrExternalLinks += 1
        dfQPPS3.loc[i, 'nbrInternalLinks'] = nbrInternalLinks
        dfQPPS3.loc[i, 'nbrExternalLinks'] = nbrExternalLinks
    

    
#on ne garde que  les enregistrement avec "body"
indexGoodBody =  dfQPPS3[(dfQPPS3.loc[dfQPPS3['Body']==""]==False)].index
dfQPPS3 = dfQPPS3.iloc[indexGoodBody]  #on ne garde que les bons enregistremments 
dfQPPS3.reset_index(inplace=True, drop=True) #on reindexe
dfQPPS3.info(verbose=True)  #12194 aucune perte
#svg en flat file
dfQPPS3.to_json("dfQPPS4.json")  #

#vider la memoire
del dfQPPS3
gc.collect()

################################################
# Traitements non effectué dans la boucle

#Relecture ############
dfQPPS4 = pd.read_json("dfQPPS4.json")
dfQPPS4.info(verbose=True) # #12194
dfQPPS4.reset_index(inplace=True, drop=True) 

#apply pour tokens
dfQPPS4['tokensTitle'] = dfQPPS4['Title'].apply(tokenizer.tokenize)
dfQPPS4['tokensDescription'] = dfQPPS4['Description'].apply(tokenizer.tokenize)
dfQPPS4['tokensH1'] = dfQPPS4['H1'].apply(tokenizer.tokenize)
dfQPPS4['tokensH2'] = dfQPPS4['H2'].apply(tokenizer.tokenize)
dfQPPS4['tokensH3'] = dfQPPS4['H3'].apply(tokenizer.tokenize)
dfQPPS4['tokensH4'] = dfQPPS4['H4'].apply(tokenizer.tokenize)
dfQPPS4['tokensH5'] = dfQPPS4['H5'].apply(tokenizer.tokenize)
dfQPPS4['tokensH6'] = dfQPPS4['H6'].apply(tokenizer.tokenize)
dfQPPS4['tokensB'] = dfQPPS4['B'].apply(tokenizer.tokenize)
dfQPPS4['tokensI'] = dfQPPS4['I'].apply(tokenizer.tokenize)
dfQPPS4['tokensEM'] = dfQPPS4['EM'].apply(tokenizer.tokenize)
dfQPPS4['tokensStrong'] = dfQPPS4['Strong'].apply(tokenizer.tokenize)
dfQPPS4['tokensBody'] = dfQPPS4['Body'].apply(tokenizer.tokenize)


#Somme des TFIDF - somme des pertinences lexicales des mots 
#dfQPPS4['sumTFIDFPath'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensPath']) #déjà calculé plus haut
dfQPPS4['sumTFIDFTitle'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensTitle'])
dfQPPS4['sumTFIDFDescription'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensDescription'])
dfQPPS4['sumTFIDFH1'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensH1'])
dfQPPS4['sumTFIDFH2'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensH2'])
dfQPPS4['sumTFIDFH3'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensH3'])
dfQPPS4['sumTFIDFH4'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensH4'])
dfQPPS4['sumTFIDFH5'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensH5'])
dfQPPS4['sumTFIDFH6'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensH6'])
dfQPPS4['sumTFIDFB'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensB'])
#fQPPS4['sumTFIDFI'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensI'])  #a virer aucun I
dfQPPS4['sumTFIDFI']= 0.0 #pour éviter plantage
dfQPPS4['sumTFIDFEM'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensEM'])
dfQPPS4['sumTFIDFStrong'] = getSumTFIDFfromDFColumn(dfQPPS4['tokensStrong'])
dfQPPS4['sumTFIDFBody'] =  getSumTFIDFfromDFColumn(dfQPPS4['tokensBody'])

dfQPPS4.to_json("dfQPPS5.json")  #Sauvegarde


del dfQPPS4
gc.collect()


###########################################################################################
#Calculons plutôt une fréquence de sumTFIDF vs la longueur de l'objet recherché 
#pour avoir "l'originalité" de l'objet : Plus c'est grand plus c'est "original" 
#on rajoute 0.01 pour éviter les divisions par zéro
###########################################################################################
#Relecture pour continuer ############
dfQPPS5 = pd.read_json("dfQPPS5.json")
dfQPPS5.info(verbose=True) # 12194  enregistrements.    
dfQPPS5.reset_index(inplace=True, drop=True) 


dfQPPS5['sumTFIDFPageFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFPage']/(x['lenTokensPage']+0.01),axis=1) 

dfQPPS5['sumTFIDFWebSiteFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFWebSite']/(x['lenTokensWebSite']+0.01),axis=1)    
dfQPPS5['sumTFIDFPathFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFPath']/(x['lenTokensPath']+0.01),axis=1) 

dfQPPS5['sumTFIDFTitleFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFTitle']/(x['lenTokensTitle']+0.01),axis=1)      
dfQPPS5['sumTFIDFDescriptionFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFDescription']/(x['lenTokensDescription']+0.01),axis=1)      
dfQPPS5['sumTFIDFH1Frequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFH1']/(x['lenTokensH1']+0.01),axis=1)      
dfQPPS5['sumTFIDFH2Frequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFH2']/(x['lenTokensH2']+0.01),axis=1)      
dfQPPS5['sumTFIDFH3Frequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFH3']/(x['lenTokensH3']+0.01),axis=1)      
dfQPPS5['sumTFIDFH4Frequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFH4']/(x['lenTokensH4']+0.01),axis=1)      
dfQPPS5['sumTFIDFH5Frequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFH5']/(x['lenTokensH5']+0.01),axis=1)      
dfQPPS5['sumTFIDFH6Frequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFH6']/(x['lenTokensH6']+0.01),axis=1)      
dfQPPS5['sumTFIDFBFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFB']/(x['lenTokensB']+0.01),axis=1)      
dfQPPS5['sumTFIDFIFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFI']/(x['lenTokensI']+0.01),axis=1)      
dfQPPS5['sumTFIDFEMFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFEM']/(x['lenTokensEM']+0.01),axis=1)      
dfQPPS5['sumTFIDFStrongFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFStrong']/(x['lenTokensStrong']+0.01),axis=1)      
dfQPPS5['sumTFIDFBodyFrequency'] = dfQPPS5.apply(lambda x : x['sumTFIDFBody']/(x['lenTokensBody']+0.01),axis=1)      


#Attention très gourmand en mémoire. !!!!!!!!!!!!!!!
dfQPPS5.to_json("dfQPPS6.json")  # sauvegarde en flat file json
#ce dernier fichier dfQPPS6.json sera utilisé pour le machine Learning.


#Sauvegarde dans un format plus léger qui sera utilisé pour le machine Learning.
#dans un prochain article.
#on ne sauvegarde que les variables qui nous intéresseront  par la suite
#Relecture pour continuer ############
dfQPPS6 = pd.read_json("dfQPPS6.json")
dfQPPS6.info(verbose=True) # 12194  enregistrements.    
dfQPPS6.reset_index(inplace=True, drop=True) 

dfQPPS7 =  dfQPPS6[['query', 'page', 'position', 'group','isHttps', 'level', 
             'lenWebSite', 'lenTokensWebSite',  'lenTokensQueryInWebSiteFrequency',  'sumTFIDFWebSiteFrequency',            
             'lenPath', 'lenTokensPath',  'lenTokensQueryInPathFrequency' , 'sumTFIDFPathFrequency',  
              'lenTitle', 'lenTokensTitle', 'lenTokensQueryInTitleFrequency', 'sumTFIDFTitleFrequency',
              'lenDescription', 'lenTokensDescription', 'lenTokensQueryInDescriptionFrequency', 'sumTFIDFDescriptionFrequency',
              'lenH1', 'lenTokensH1', 'lenTokensQueryInH1Frequency' ,  'sumTFIDFH1Frequency',        
              'lenH2', 'lenTokensH2',  'lenTokensQueryInH2Frequency' ,  'sumTFIDFH2Frequency',          
              'lenH3', 'lenTokensH3', 'lenTokensQueryInH3Frequency' , 'sumTFIDFH3Frequency',
              'lenH4',  'lenTokensH4','lenTokensQueryInH4Frequency', 'sumTFIDFH4Frequency', 
              'lenH5', 'lenTokensH5', 'lenTokensQueryInH5Frequency', 'sumTFIDFH5Frequency', 
              'lenH6', 'lenTokensH6', 'lenTokensQueryInH6Frequency', 'sumTFIDFH6Frequency', 
              'lenB', 'lenTokensB', 'lenTokensQueryInBFrequency', 'sumTFIDFBFrequency', 
              'lenEM', 'lenTokensEM', 'lenTokensQueryInEMFrequency', 'sumTFIDFEMFrequency', 
              'lenStrong', 'lenTokensStrong', 'lenTokensQueryInStrongFrequency', 'sumTFIDFStrongFrequency', 
              'lenBody', 'lenTokensBody', 'lenTokensQueryInBodyFrequency', 'sumTFIDFBodyFrequency', 
              'elapsedTime', 'nbrInternalLinks', 'nbrExternalLinks' ]] 

dfQPPS7.to_csv("dfQPPS7.csv", sep=",", encoding='utf-8', index=False) 
dfQPPS7FR.to_csv("dfQPPS7FR.csv", sep=";", encoding='utf-8', index=False) #séparateur ; 


##########################################################################
# MERCI pour votre attention !
##########################################################################
#on reste dans l'IDE
#if __name__ == '__main__':
#  main()

