import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import classification_report, auc, plot_roc_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from PIL import Image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pickle


data = np.load("./data/phishing.npz")
x_train_, y_train_ = data['X_train'], data['y_train']
data_test = np.load("./data/phishing_test.npz")
x_test_, y_test_ = data['X_test'], data['y_test']

# alphabet = string.ascii_lowercase + string.digits + "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
# reverse_dictionary = {}
# for i, c in enumerate(alphabet):
#     reverse_dictionary[i + 1] = c

# x_train = []
# for url in x_train_:
#     this_url = ""
#     for position in url:
#         this_index = np.argmax(position)
#         if this_index != 0:
#             this_url += reverse_dictionary[this_index]
#     x_train.append(this_url)

# x_test = []
# for url in x_test_:
#     this_url = ""
#     for position in url:
#         this_index = np.argmax(position)
#         if this_index != 0:
#             this_url += reverse_dictionary[this_index]
#
#     x_test.append(this_url)


y_train = []
for label in y_train_:
    y_train.append(label[1])


y_test = []
for label in y_test_:
    y_test.append(label[1])

# total = x_train + x_test
# print(len(total))
# total_label = y_train + y_test
# tokenizer = RegexpTokenizer(r'[A-Za-z]+')
# print(tokenizer.tokenize(x_test[0]))
#
# df = DataFrame()
# df['URL'] = total
# df['label'] = total_label
# df['text_tokenized'] = df.URL.map(lambda t: tokenizer.tokenize(t))
# stemmer = SnowballStemmer("english")
# df['text_stemmed'] = df['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
# df['text_sent'] = df['text_stemmed'].map(lambda l: ' '.join(l))
# cv = CountVectorizer()
# feature = cv.fit_transform(df.text_sent)
#
# trainX, testX, trainY, testY = train_test_split(feature, df['label'], shuffle=False)
# print(trainX.shape, trainY.shape)

# y_train = []
# for label in y_train_:
#     y_train.append(label[1])
# #
# y_test = []
# for label in y_test_:
#     y_test.append(label[1])
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#
x_train = np.reshape(x_train_, (40000, 200*67))
x_test = np.reshape(x_test_, (10000, 200*67))
trainX, trainY = x_train, y_train
testX, testY = x_test, y_test

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
# mnb = MultinomialNB()
# mnb.fit(trainX, trainY)
# print("mnb: ", mnb.score(testX, testY))

svc = LinearSVC()
svc.fit(trainX, trainY)
print("svc", svc.score(testX, testY))

lr = LogisticRegression()
lr.fit(trainX, trainY)
print("lr", lr.score(testX, testY))

rf = RandomForestClassifier()
rf.fit(trainX, trainY)
print("rf", rf.score(testX, testY))

from sklearn.base import BaseEstimator, TransformerMixin

def results(name: str, model: BaseEstimator) -> None:
    preds = model.predict(testX)
    fpr, tpr, thresholds = roc_curve(testY, preds)
    roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    #
    # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()

    print(name + ":", model.score(testX, testY), recall_score(testY, preds), fpr, precision_score(testY, preds), f1_score(testY, preds), roc_auc_score(testY, preds))
    # print(name + "auc: %.4f" % roc_auc_score(testY, preds))
    # print(classification_report(testY, preds))
    # labels = ['Good', 'Bad']
    #
    # conf_matrix = confusion_matrix(testY, preds)
    #
    # font = {'family' : 'normal',
    #         'size'   : 14}
    #
    # plt.rc('font', **font)
    # plt.figure(figsize= (10,6))
    # sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap='Blues')
    # plt.title("Confusion Matrix for " + name)
    # plt.ylabel('True Class')
    # plt.xlabel('Predicted Class')

svc_disp = plot_roc_curve(svc, testX, testY)
# mnb_disp = plot_roc_curve(mnb, testX, testY, ax=svc_disp.ax_)
lr_disp = plot_roc_curve(lr, testX, testY, ax=svc_disp.ax_)
rf_disp = plot_roc_curve(rf, testX, testY, ax=svc_disp.ax_)
plt.xlim([-0.05, 0.6])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([0.3, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
# rf_disp.figure_.suptitle("ROC curve")
results("SVC", svc)
# results("nb", mnb)
results("lr", lr)
results("rf", rf)
plt.show()
svc_disp.figure_.savefig('roc_curve_ml.eps', dpi=1200, format='eps')