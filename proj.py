# Import of libs
#--------------------------#
import pandas as pd
import numpy as np
import skimpy
import spacy
import json
from typing import Dict
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Chargement des données
#-----------------------#

df = pd.read_csv("C:/Users/PC/Desktop/HRFlow - Recommandation d'emploi basée sur le comportement/x_train_Meacfjr.csv")


with open('job_listings.json') as f:
    data = json.load(f)


# fun: Génération d'un VRAI nested dict avec les des clefs plus logiques 
# ------------------------------ #
def generate_proper_dict(data: Dict) -> Dict:
    jobs_struct={}
    for job_id, text in data.items():
        job_info={
            "job_id":job_id,
            "title":None,
            "summary":None,
            "description":None,
            "languages":None,
            "certifications":None,
            "skills":None,
            "courses":None,
            "tasks":None
            }
            
        sections = ["TITLE", "SUMMARY", "DESCRIPTION", "LANGUAGES", 
                    "CERTIFICATIONS", "SKILLS", "COURSES", "TASKS"]
        pattern = rf"\b({'|'.join(sections)}):?\b"
        
        matches = list(re.finditer(pattern, text))
        for i, match in enumerate(matches):
            section_name = match.group(1)  # Extract section name
            start = match.end()  # Start of this section’s content
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)  # End of section
            
            section_text = text[start:end].strip()
            if section_name == "TITLE":
                job_info['title'] = section_text
            elif section_name == "SUMMARY":
                job_info['summary'] = section_text
            elif section_name == "DESCRIPTION":
                job_info["description"] = section_text
            elif section_name == "LANGUAGES" :
                job_info['languages'] = section_text
            elif section_name == "CERTIFICATIONS":
                job_info["certifications"] = section_text
            elif section_name == "SKILLS" :
                job_info['skills'] = section_text
            elif section_name == "COURSES":
                job_info["courses"] = section_text
            elif section_name == "TASKS" :
                job_info['tasks'] = section_text
                
        jobs_struct[job_id]=job_info
    return jobs_struct
# end fun
# ------------------------------ #

# Génération données structurées
# ------------------- #
structured_data = generate_proper_dict(data)

# Création d'un dataframe basé sur le dictionnaire plus propre
#----------------------------------------#
emploi = pd.DataFrame(structured_data.values())

# Nettoyage et application de traitement linguistique  sur emploi *
#-------------------#

# Nettoyage des données 
#------------------------#

emploi.info()

# Visualisation des offres sans description 

missing_descriptions = emploi[emploi["description"].isna()]


# Initialisation des moteurs nlp
#---------------------------#
nlp = spacy.load("fr_core_news_md")
lemmatizer = WordNetLemmatizer()

# fun: Lemmatisation des cellules textuelles d'un dataframe
#---------------------------#
def lemmatize_text(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Join the lemmatized words back into a string
    lemmatized_text = ' '.join(lemmatized_words)

    return lemmatized_text




# Application de la fonction
# En commentaire avant de trouver uen stratégie de gestion des null values ou Nan
#-----------------------------#*
"""
emploi = emploi.applymap(str)

emploi[["lemm_title", "lemm_summary", "lem_description"]] = emploi[["title","summary", "description"]].apply(lemmatize_text)
"""
        
