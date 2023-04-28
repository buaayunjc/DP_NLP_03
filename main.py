# — coding: utf-8 –
import math
import jieba
import os  # 用于处理文件路径
import random
import numpy as np
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import matplotlib
import pyLDAvis.gensim



def read_novel(path):  # 读取语料内容
    with open('stopwords.txt', 'r', encoding='utf-8',errors='ignore') as f:
        stop = set([line.strip() for line in f])

    content_train = []
    content_test = []
    names = os.listdir(path)
    for name in names:
            con_temp = []
            novel_name = path + '/' + name
            with open(novel_name, 'r', encoding='ANSI',errors="ignore") as f:
                con = f.read()
                con = content_deal(con)
                con = jieba.lcut(con)
                # 词模型
                # con = [word for word in con if word not in stop]
                # 字模型
                con = [char for word in con for char in word]
                con = [word for word in con if word not in stop]
                con_list = list(con)
                pos = int(len(con)//13)
                for i in range(13):
                    con_temp = con_temp + con_list[i*pos:i*pos+500]
                content_train.append(con_temp)
                con_temp = []
                for i in range(13):
                    con_temp = con_temp + con_list[i*pos+501:i*pos+1000]
                content_test.append(con_temp)
            f.close()
    return content_train,content_test, names


def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』','=','（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    return content


def coherence(num_topics):
    ldamodel = models.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    ldacm = CoherenceModel(model=ldamodel, texts=train_txt, dictionary=dictionary, coherence='c_v')
    print(ldacm.get_coherence())
    return ldacm.get_coherence()


#计算困惑度
def perplexity(num_topics):
    ldamodel = models.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=15))
    print(ldamodel.log_perplexity(corpus))
    return ldamodel.log_perplexity(corpus)


if __name__ == '__main__':
    [train_txt,test_txt, labels] = read_novel("金庸小说集")
    dictionary = corpora.Dictionary(train_txt)
    dictionary.filter_n_most_frequent(100)

    x = range(7, 19)
    # y = [perplexity(i) for i in x]  #如果想用困惑度就选这个
    y = [coherence(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('主题数目')
    plt.ylabel('coherence大小')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('主题-coherence变化情况')
    plt.show()

    corpus = [dictionary.doc2bow(text) for text in train_txt]  # 第几个单词出现了几次
    ldamodel = models.LdaModel(corpus, num_topics=17, id2word=dictionary, passes=30, random_state=1)  # 分为10个主题

    import pyLDAvis.gensim
    import pyLDAvis
    pyLDAvis.enable_notebook()
    data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.save_html(data, '/data/topic.html')

