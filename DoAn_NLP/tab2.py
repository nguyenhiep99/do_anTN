import pandas as pd
import _pickle as cPickle
from tab1 import read_json, split_words
import pickle
import openpyxl

DATA_TRAIN_JSON = 'data/train/data_train.json'
model_path = 'models'

def build_dataset2(data):
    contents = []
    i = 0
    for d in data:
        i += 1
        words = (split_words(d['Nội dung góp ý']))
        if (len(words) < 0):
            continue
        sentence = " ".join(words)
        contents.append(sentence)
    return contents

def write_excel_to_json1(pathExcel, pathJson):
    df = pd.read_excel(pathExcel, skiprows=8)
    df.to_json(pathJson, orient="records")

def buld_text(text):
    words = split_words(text)
    if (len(words) <= 0):
        return ""
    sentence = " ".join(words)
    return sentence

def predict_text1(model_name, text):
    loaded_model = cPickle.load(open(model_path + '/' + model_name, 'rb'))
    contents = buld_text(text)
    print(contents)
    Tfidf_vect = pickle.load(open('tfidf/' + model_name, 'rb'))
    Train_X_Tfidf = Tfidf_vect.transform([contents])
    label = (loaded_model.predict(Train_X_Tfidf))
    return label[0]

def create_excel(DATA_TRAIN_PATH, filename):
    loaded_model = cPickle.load(open(model_path + '/' + filename, 'rb'))
    write_excel_to_json1(DATA_TRAIN_PATH, DATA_TRAIN_JSON)
    data = read_json(DATA_TRAIN_JSON)
    contents = build_dataset2(data)
    Tfidf_vect = pickle.load(open('tfidf/' + filename, 'rb'))
    Train_X_Tfidf = Tfidf_vect.transform(contents)
    labels = (loaded_model.predict(Train_X_Tfidf))
    k, l, m = 0, 0, 0
    for i in range(len(labels)):
        if (labels[i] == 'Pos'):
            k += 1
        elif (labels[i] == 'Neg'):
            l += 1
        elif (labels[i] == 'Non'):
            m += 1
    return labels, k, l,m
def save_file(export, DATA_TRAIN_PATH, labels):
    wb = openpyxl.load_workbook(DATA_TRAIN_PATH)
    ws = wb.active
    for i, v in enumerate(labels):
        ws.cell(row=10 + i, column=8).value = v
    wb.save(export[0])
