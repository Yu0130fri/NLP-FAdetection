#====================================================
#　形態素解析のためのmodel	
#====================================================

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import word2vec
import MeCab
import mojimoji
import numpy as np
import pandas as pd
from pyknp import Juman


#====================================================
#　原形を出力	
#====================================================

def juman_text_to_word_genkei(text, stopword_pass="./stopwords/Japanese.txt"):
    """JUMAN++を使い、名詞・動詞・形容詞の原型をリストに格納する関数"""
    #stopwordのリスト作成
    stopword_list =[]
    with open(stopword_pass, "r") as file:
        lines = file.readlines()
            
        stopword_list = [stopword for stopword in lines if stopword.strip()]
        
    #mojimojiで全角数字、英字を半角に統一。大文字もすべて小文字に統一する
    try:
        text = mojimoji.zen_to_han(text, kana=False).lower()
    except:
        pass
    
    jumanpp = Juman()
    #形態素解析する
    result = jumanpp.analysis(text)
    #分類された各形態素ごとに見ていく
    morph_list = result.mrph_list()
    
    basic_words =[]
    for morph in morph_list:
        #記号なら何もせず続ける
        if morph.bunrui == "記号":
            continue
        elif morph.hinsi =="名詞" and morph.genkei not in stopword_list:
            basic_words.append(morph.genkei)
        elif morph.hinsi in ["動詞", "形容詞"] and morph.genkei not in stopword_list:
            basic_words.append(morph.genkei)
            
    #basic_wordsのリストを半角スペースでまとめておく
    basic_words = " ".join(basic_words)
    return basic_words



#====================================================
#　分かち書きを出力	
#====================================================

def juman_text_to_word_wakati(text, stopword_pass="./stopwords/Japanese.txt"):
    """JUMAN++を使い、名詞・動詞・形容詞の原型をリストに格納する関数"""
    #stopwordのリスト作成
    stopword_list =[]
    with open(stopword_pass, "r") as file:
        lines = file.readlines()
            
        stopword_list = [stopword for stopword in lines if stopword.strip()]
        
    #mojimojiで全角数字、英字を半角に統一。大文字もすべて小文字に統一する
    #0305修正
    text = mojimoji.zen_to_han(text, kana=False).lower()

    jumanpp = Juman()
    #形態素解析する
    result = jumanpp.analysis(text)
    #分類された各形態素ごとに見ていく
    morph_list = result.mrph_list()
    
    basic_words =[]
    for morph in morph_list:
        basic_words.append(morph.midasi)
            
    #basic_wordsのリストを半角スペースでまとめておく
    basic_words = " ".join(basic_words)
    return basic_words


