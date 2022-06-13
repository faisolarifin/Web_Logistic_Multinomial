import os
import logging
import math
import time
import pickle
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, redirect, url_for, abort, session

###=>>FLASK ROOT SETTING
project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')

app = Flask(__name__, template_folder=template_path, static_folder=static_path)
app.secret_key = 'ini kunci rahasia'

#memanggil data csv
data = pd.read_csv('data/datafixx.csv', sep=';')
#PREDIKAT
data['PREDIKAT'].value_counts()
data['PREDIKAT id'] = data['PREDIKAT'].factorize()[0]
kategori_id_data = data[['PREDIKAT', 'PREDIKAT id']].drop_duplicates().sort_values('PREDIKAT id')
kategori_to_id = dict(kategori_id_data.values)
PREDIKAT = dict(kategori_id_data[['PREDIKAT id', 'PREDIKAT']].values)

def termFrequency():
    ds = pd.read_csv('data/dataset term frequency.csv', sep=';')
    jenisKelamin = zip(ds['kategori'][:2],ds['term frequency'][:2])
    daerahAsal = zip(ds['kategori'][2:9],ds['term frequency'][2:9])
    jalurMasuk = zip(ds['kategori'][9:],ds['term frequency'][9:])
    return jenisKelamin, daerahAsal, jalurMasuk

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        JENIS_KELAMIN = request.form['gender']
        ASAL_DAERAH = request.form['daerah']
        JALUR_MASUK = request.form['jalur']
        IP1 = request.form['ip1']
        IP2 = request.form['ip2']
        IP3 = request.form['ip3']
        IP4 = request.form['ip4']
        dictData = {
            'JENIS_KELAMIN' : [JENIS_KELAMIN],
            'ASAL_DAERAH': [ASAL_DAERAH],
            'JALUR_MASUK' : [JALUR_MASUK],
            'IP 1' : [IP1], 
            'IP 2' : [IP2], 
            'IP 3' : [IP3], 
            'IP 4' : [IP4]
        }
        dataPredict = pd.DataFrame(data = dictData)

        if os.path.exists('model/model_lr.pkl'):
            model = pickle.load(open('model/model_lr.pkl','rb'))
            pred = model.predict(dataPredict)
            predictRes = PREDIKAT[pred[-1]]

            return render_template('index.html', s=True, data=dictData, predict=predictRes, termFreq=termFrequency())

    return render_template('index.html', predictStatus=False, termFreq=termFrequency())

#run flask server
if __name__ == '__main__':
#   logging.basicConfig(filename='static/error.log',level=logging.DEBUG)
  app.run(host='0.0.0.0',port=80, debug=True)