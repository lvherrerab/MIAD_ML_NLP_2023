from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import pandas as pd
import sys
import os
import joblib

pipeline = joblib.load(os.path.dirname(__file__) + '/car_price_model.pkl') 



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app,
    version='1.0',
    title='Car Price Prediction API',
    description='Car Price Prediction API')

ns = api.namespace('predict',
                   description='Car Price Predictor')

car = api.model('Car', {
    'Year': fields.Integer(required=True, description='Car manufacture year'),
    'Mileage': fields.Integer(required=True, description='Car mileage'),
    'State': fields.String(required=True, description='Car registration state'),
    'Make': fields.String(required=True, description='Car make'),
    'Model': fields.String(required=True, description='Car model')
})

price_fields = api.model('Price', {
    'predicted_price': fields.Float
})


@ns.route('/')
class CarPriceApi(Resource):

    @api.expect(car)
    @api.marshal_with(price_fields)
    def post(self):
        data = api.payload

        year = data['Year']
        mileage = data['Mileage']
        state = data['State']
        make = data['Make']
        model = data['Model']

        # Crear un DataFrame con los datos ingresados
        input_data = pd.DataFrame([[year, mileage, state, make, model]], columns=["Year", "Mileage", "State", "Make", "Model"])

        # Utilizar el preprocesador y el modelo para predecir el precio del automóvil
        price_prediction = pipeline.predict(input_data)[0]

        return {"predicted_price": price_prediction}, 200



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
