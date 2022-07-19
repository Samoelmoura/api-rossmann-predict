import pandas as pd
from rossmann import Rossmann
from flask import Flask, request, Response
import os

# initialize API
app = Flask(__name__)

# create endpoint
@app.route('/predict', methods=['POST'])

def rossmann_predict():
    df_raw_json = request.get_json()

    if df_raw_json:
        if isinstance(df_raw_json, dict):
            df_raw = pd.DataFrame(df_raw_json, index=[0])

        else:
            df_raw = pd.DataFrame(df_raw_json, columns=df_raw_json[0].keys())

        # instantiate rossmann class
        papeline = Rossmann()

        # data cleaning
        df = papeline.data_cleaning(df_raw)

        # feature engineering
        df = papeline.feature_engineering(df)

        # data filtering
        df = papeline.data_filtering(df)

        # data preparation
        df = papeline.data_preparation(df)

        # predicting
        df = papeline.get_predictions(df, df_raw)

        # return df
        df_json = df.to_json(orient='records')
        return df_json

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == "__main__":
    # starting flask
    # port = os.environ.get('PORT', 5000) # Heroku port
    port = 5000 # local port
    app.run(host='0.0.0.0', port=port)
