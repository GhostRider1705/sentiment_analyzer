# -*- coding: utf-8 -*-
"""Pre-processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u7IC_5uppkvjYCGPPLfWm5ZDk-BpK6fv
"""
from nltk.stem import PorterStemmer 
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
from nltk import download
download('wordnet')
download('averaged_perceptron_tagger')
download('punkt')

import os
os.system('pip install contractions')
os.system('pip install symspellpy')

import contractions
import pkg_resources
from symspellpy.symspellpy import SymSpell , Verbosity

#Load Dictionary

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

def pre_processing(text):
  text = (contractions.fix(text))  
  suggestions = sym_spell.lookup_compound(text,max_edit_distance=2)
  # x=list(map(''.join ,[lemmatizer.lemmatize(ps.stem(suggestion.term)) for suggestion in suggestions])) 
  x=list(map(''.join ,[lemmatizer.lemmatize(suggestion.term) for suggestion in suggestions]))
  return x[0]


