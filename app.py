import dash
from dash import dcc
from dash import html
import os
import numpy as np
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import spacy
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

def ltokenize(asent):
    doc = nlp(asent)
    for token in doc:
        print(token, token.pos_, token.dep_)
    return doc

def get_tokens(asent, atoken2idx, maxlen):
    ldoc = ltokenize(asent)
    # get max token and tag length
    ltokens = [str(x).lower() for x in ldoc]
    itokens = []
    # Pad tokens (X var)
    for s in ltokens:
        if s in atoken2idx:
            itokens.append(atoken2idx[s])
        else:
            itokens.append(atoken2idx['<UNKNOWN>'])
    pad_tokens = pad_sequences([itokens], maxlen=maxlen, dtype='int32', padding='post', value=atoken2idx['<PAD>'])

    return pad_tokens, ltokens

def onehot2arr(ay):
    llist = []
    for sent in ay:
        llist = llist + np.argmax(sent, axis = 1).tolist()
    return np.array(llist)

app.layout = html.Div([
    html.H4("Введите предложение"),
    html.Div([
        "Предложение: ",
        dcc.Input(id='my-input', value='Иоанн Павел II внёс некоторые изменения в правила проведения конклавов.', type='text', style={'width': '100vh'})
    ]),
    html.Br(),
    html.Button('Распознать сущности', id='submit'),
    html.Br(),
    html.Br(),
    html.Div(id='my-output'),

])

nlp = spacy.load("ru_core_news_sm")
model_bilstm_lstm = tf.keras.models.load_model('z:\\ИТЦ ФАС\\Обучение\\Innopolis Data Science\\From Linux\\innopolis\\DZ_diplom\\model_bilstm_lstm.h5')

with open("z:\\ИТЦ ФАС\\Обучение\\Innopolis Data Science\\From Linux\\innopolis\\DZ_diplom\\token2idx.json", "r", encoding="utf-8") as fp:
    token2idx = json.load(fp)
    fp.close()

with open("z:\\ИТЦ ФАС\\Обучение\\Innopolis Data Science\\From Linux\\innopolis\\DZ_diplom\\idx2tag.json", "r", encoding="utf-8") as fp:
    idx2tag = json.load(fp)
    fp.close()

@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input('submit', 'n_clicks')],
    [State('my-input', 'value')]
)
def update_output_div(n_clicks, value):
    if (n_clicks):
        ftokens, fltokens = get_tokens(value, token2idx, 59)
        y_pred = model_bilstm_lstm.predict(ftokens)
        y_preda = onehot2arr(y_pred)
        st = ''
        for i in range(len(fltokens)):
            st += fltokens[i] + ': [' + idx2tag[str(y_preda[i])] + '],  '
        return 'Разметка: {}'.format(st)


if __name__ == '__main__':
    app.run_server(debug=True)