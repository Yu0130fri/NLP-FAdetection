from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import word2vec
import MeCab
import mojimoji
import numpy as np
import pandas as pd
from pyknp import Juman
#次元圧縮によって計算量を減らすために主成分分析をかけるため
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from models import juman_text_to_word

test_txt = pd.read_csv("AC作業_Rawdata",header=0,encoding="cp932",sep="\t")


test_txt_adjust = test_txt[["RMID", "SC10t1", "SC10t2", "SC10t3", "SC10t", "SC10t1"]]

#一旦絵文字が出てきた場合は空白に置換
import re
test_txt.replace("&#[\d]{4,6}"," ",regex=True,inplace=True)


#整形
test_txt.replace("\n","",inplace=True,regex=True)
test_txt.replace("\d","",inplace=True,regex=True)

#口コミの列を形態素解析していく
#一旦Q2FAで検証
#test_txt["SC10t1_genkei"] = test_txt["SC10t1"][:1000].map(juman_text_to_word_genkei)
test_txt["SC10t1_wakati"] = test_txt["SC10t1"][:3000].map(juman_text_to_word.juman_text_to_word_wakati)

sample_corpus = test_txt["SC10t1_wakati"]
# 出現する単語のカウントを特徴量にする手法になります。
# 出現した単語を純粋にカウントします。
vec = CountVectorizer()
#fit_transformで文書行列を取得
vec_fit_transform = vec.fit_transform(sample_corpus)

print('Bag of Words')
#カラムに全ての単語の種類を提示、インデックスは各文章を提示
display(pd.DataFrame(vec_fit_transform.toarray(), columns=vec.get_feature_names(), index=sample_corpus))

print('tfidf')
# skleanの中のTfidfTransformerでインスタンス化して利用する
tfidf = TfidfTransformer()

model_tfidf = tfidf.fit_transform(vec_fit_transform)

display(pd.DataFrame(vec_fit_transform.toarray(), columns=vec.get_feature_names(), index=sample_corpus))