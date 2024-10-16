from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import joblib

df_pos = pd.read_csv('/home/luizaperez/Documentos/cell-immunity-prediction/vetores/vetores_positivos.csv')
df_neg = pd.read_csv('/home/luizaperez/Documentos/cell-immunity-prediction/vetores/vetores_negativos.csv')
df_all = pd.concat([df_pos, df_neg])

df_response = df_all[df_all['Response'] == 'cytotoxicity']

if df_response.shape[0] > 10000:
    df_response = df_response.sample(10000)
    
#preparar x e y (features and labels)
X = list(df_response['Vetor médio'].map(json.loads))
y = df_response['Result'] == 'positive'

print('positive:', list(df_response['Result']).count('positive'))
print('negative:', list(df_response['Result']).count('negative'))
    
if list(df_response['Result']).count('positive') >= 100 and list(df_response['Result']).count('negative') >= 100:
    print('citotoxicity')
    
    #split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    #balanceamento
    X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
    X_test, y_test = RandomUnderSampler().fit_resample(X_test, y_test)
    
    #treinar modelo
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    #predições
    y_pred = model.predict(X_test)
    
    #printar resultado
    report = classification_report(y_test, y_pred)
    print(report)
    
    print('------------------------------------------------------')
    
joblib.dump(model, 'train_model_cytotoxicity.pkl')
