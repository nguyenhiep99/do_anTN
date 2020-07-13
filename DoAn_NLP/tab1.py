import pandas as pd
import json
from pyvi import ViTokenizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import _pickle as cPickle
import pickle
import re
import os

# cac duong dan
DATA_TRAIN_JSON = 'data/train/data_train.json'
DATA_TEST_JSON = 'data/train/data_test.json'
STOP_WORDS = 'stopwords.txt'
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''
model_path = 'models'

# chuyen excel sang json
def write_excel_to_json(pathExcel, pathJson):
    df = pd.read_excel(pathExcel)
    df.to_json(pathJson, orient="records")

# doc json
def read_json(pathJson):
    with open(pathJson, encoding="utf-8") as f:
        s = json.load(f)
    return s

# tach tu tieng  viet and chuan hoa tu
def segmentation(text):
        text =re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
        return ViTokenizer.tokenize(text)

#chuẩn hóa thành kí tự thường
def split_words(text):
        _text = segmentation(text)
        try:
            r = [x.strip(SPECIAL_CHARACTER).lower() for x in _text.split()]
            return [i for i in r if i]
        except TypeError:
            return []
# tao content and labels
def build_dataset(data):
        contents = []
        labels = []
        i = 0
        for d in data:
            i += 1
            words = (split_words(d['Nội dung góp ý']))
            if (len(words) < 0):
              continue
            # contents.append(" ".join(word))
            sentence = " ".join(words)
            contents.append(sentence)
            labels.append(d['Đánh giá'])
        return contents, labels
# training
def training(algorithm, features_train, labels_train, features_test, labels_test):
        algorithm.fit(features_train, labels_train)
        scores = algorithm.score(features_test, labels_test)
        return scores
# save model
def save_model(filePath, obj, data =None):

    with open(filePath, 'w') as outfile:
        json.dump(data, outfile)
    outfile = open(model_path + '/' + filePath.replace(' ', '_') + '.pkl', 'wb')
    fastPickler = cPickle.Pickler(outfile, -1)
    fastPickler.fast = 1
    fastPickler.dump(obj)
    outfile.close()

def traning1(DATA_TRAIN_PATH, DATA_TEST_PATH,model_name,  algorithm_index, algorithm_config):
    write_excel_to_json(DATA_TRAIN_PATH, DATA_TRAIN_JSON)
    data_train = read_json(DATA_TRAIN_JSON)
    write_excel_to_json(DATA_TEST_PATH, DATA_TEST_JSON)
    data_test = read_json(DATA_TEST_JSON)
    X_train, Y_train = build_dataset(data_train)
    X_test, Y_test = build_dataset(data_test)
    Tfidf_vect = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.3)
    Tfidf_vect.fit(X_train)

    pickle.dump(Tfidf_vect, open('tfidf/' + model_name.replace(' ', '_') + '.pkl', 'wb'))
    Train_X_Tfidf = Tfidf_vect.transform(X_train)
    Test_X_Tfidf = Tfidf_vect.transform(X_test)

    if algorithm_index == 0:
        Naive = MultinomialNB(alpha=algorithm_config.alpha)
        result = training(Naive, Train_X_Tfidf, Y_train, Test_X_Tfidf, Y_test)
        save_model(model_name, obj=Naive)
        return str(round((result * 100), 2)) + ' %'

    elif algorithm_index == 1:
        SVM = SVC(kernel= algorithm_config.kernel,C=algorithm_config.C, random_state=0)
        result = training(SVM, Train_X_Tfidf, Y_train, Test_X_Tfidf, Y_test)
        save_model(model_name, obj=SVM)
        return "Độ chính xác: " + str(round((result * 100), 2)) + ' %'









