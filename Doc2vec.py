#-*- coding: UTF-8 -*-
import gensim
import numpy as np
import jieba
import sys, json
import requests
import pymysql
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

TaggededDocument = gensim.models.doc2vec.TaggedDocument


# 데이터 전처리
def get_corpus():
    with open("/var/www/html/document.txt", 'r') as doc:
        docs = doc.readlines()
    train_docs = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        length = len(word_list)
        word_list[length - 1] = word_list[length - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        train_docs.append(document)
    return train_docs


# 모델들을 학습시킴
def Doc2vec_train(x_train, size=200):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('/var/www/html/music_train_model')
    # 닥투백 모델 저장시킴
    return model_dm


# 학습된 닥투백 모델을 사용함
def Model_test():
    model_dm = Doc2Vec.load("music_train_model")
    text_test = musictitle
    text_cut = jieba.cut(text_test)
    text_raw = []
    for i in list(text_cut):
        text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims


if __name__ == '__main__':
    # print("33")
    # 데이터 전처리
    x_train = get_corpus()
    # 독투백 모델을 학습시킬때 주석을 풀것!
    # model_dm = Doc2vec_train(x_train)
    sims = Model_test()
    # 모델을 불러옴

    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        whatis_musictitle=words.split(';')
        print(whatis_musictitle[0])
        # 유사한 노래만 불러