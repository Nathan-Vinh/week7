from dash import callback, ctx, html
from dash.dependencies import Input, Output
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
import pandas as pd
#import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences

@callback(
    [
        Output('return-paper', 'children')
    ],
    [
        Input('tf-button', 'n_clicks'),
        Input('sk-button', 'n_clicks'),
        Input('textinput', 'value')
    ]
)
def sklearnmodel(tfb, skb, text):
    df = pd.read_csv('spam_clean.csv')

    X = df[['text_lem']]
    y = df['0']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=y)

    tfidf_vect = TfidfVectorizer(analyzer=tostring)
    tfidf_vect_fit = tfidf_vect.fit(X_train['text_lem'])

    tfidf_train = tfidf_vect_fit.transform(X_train['text_lem'])
    tfidf_test = tfidf_vect_fit.transform(X_test['text_lem'])

    if 'sk-button' == ctx.triggered_id:
        # Create a perceptron object
        ppn = Perceptron(random_state=0)


        # Train the perceptron
        ppn.fit(tfidf_train, y_train)


        input_no_punct = remove_punct(text)
        input_tokenized = toknize(input_no_punct)
        input_no_sw = remove_stopwords(input_tokenized)
        input_lem = lemmatizing(input_no_sw)

        input_vec = tfidf_vect_fit.transform([input_lem])

        result = ppn.predict(input_vec)[0]

        if result == 'ham':
            return [html.H4(f"With Sklearn model, your message is a {result}", style={'color':'green'})]
        else:
            return [html.H4(f"With Sklearn model, your message is a {result}", style={'color':'red'})]


    elif 'tf-button' == ctx.triggered_id:
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=100),
        #     tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        #     tf.keras.layers.MaxPooling1D(pool_size=2),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(units=10, activation='relu'),
        #     tf.keras.layers.Dense(units=1, activation='relu')
        # ])

        # model.compile(loss='binary_crossentropy',
        #               optimizer='adam', metrics=['accuracy'])
        # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        # history = model.fit(tfidf_train, y_train, epochs=10,
        #                     validation_data=(tfidf_test, y_test), callback=[callback])

        # result = model.predict()

        result= "model not loaded"

        if result == 'ham':
            return [html.H4(f"With TensorFlow model, your message is a {result}", style={'color': 'green'})]
        else:
            return [html.H4(f"With TensorFlow model, your message is a {result}", style={'color': 'red'})]
