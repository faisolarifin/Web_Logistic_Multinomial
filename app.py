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
#Extract data
X = data.iloc[:,2:9].values
y = data.iloc[:,-1].values
# membagi data set menggunakan sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

def getEquation():
    def Fraction(equ, num, den, res):
        return f'<div class="All"><div class="Sum">{equ} =</div><span class="Fraction"><span class="Numerator">{num}</span><span class="Denominator">{den}</span></span><div class="Sum"> = {res}</div></div>'
    myEq = []
    logit_model=sm.MNLogit(y_train,sm.add_constant(X_train))
    result=logit_model.fit()
    df = pd.read_html(result.summary().tables[1].as_html())[0]
    gi=1
    resMath = []
    for i, (var, coef) in enumerate(zip(df['y=1'].values, df['coef'].values)):
        if i==0 or i==8:
            eq =f'g<sub>{gi}</sub>(x) = '
            calculate = 0
        if i !=8:
            coef = float(coef)
            if coef >= 0:
                eq += f'+ {coef}'
            else:
                eq += f' {coef}'
            if var != 'const':
                eq += f' {var} '
            calculate += coef
        if i==len(df)//2-1 or i==len(df)-1:
            gi += 1
            eq += ' = {:.3f}'.format(calculate)
            myEq.append(eq)
            resMath.append(calculate)

    resMath.append(
        math.exp(resMath[0])/(1 + math.exp(resMath[0] + math.exp(resMath[1])))
    )
    resMath.append(
        math.exp(resMath[1])/(1 + math.exp(resMath[0] + math.exp(resMath[1])))
    )
    resMath.append(
        1/(1 + math.exp(resMath[0] + math.exp(resMath[1])))
    )
    myEq.append(Fraction(
        '&pi;<sub>1</sub>(x) ',
        'exp({:.3f})'.format(resMath[0]),
        '1 + exp({:.3f}) + exp({:.3f})'.format(resMath[0], resMath[1]),
        resMath[2]
    ))
    myEq.append(Fraction(
        '&pi;<sub>2</sub>(x) ',
        'exp({:.3f})'.format(resMath[1]),
        '1 + exp({:.3f}) + exp({:.3f})'.format(resMath[0], resMath[1]),
        resMath[3]
    ))
    myEq.append(Fraction(
        '&pi;<sub>3</sub>(x) ',
        '1',
        '1 + exp({:.3f}) + exp({:.3f})'.format(resMath[0], resMath[1]),
        resMath[4]
    ))
    return myEq

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

            return render_template('index.html', s=True, data=dictData, predict=predictRes, eq=getEquation(), termFreq=termFrequency())

    return render_template('index.html', predictStatus=False, eq=getEquation(), termFreq=termFrequency())


if __name__ == '__main__':
#   logging.basicConfig(filename='static/error.log',level=logging.DEBUG)
  app.run(host='0.0.0.0',port=8000, debug=True)