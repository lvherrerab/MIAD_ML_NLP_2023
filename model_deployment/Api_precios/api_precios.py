from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import pandas as pd
import joblib

pipeline = joblib.load("/Users/LinaH/Documents/GitHub/MIAD_ML_NLP_2023/model_deployment/Api_precios/precios_ante.pkl")

app = Flask(__name__) # Importa y crea una instancia de la clase Flask para crear una aplicación web. Flask es un framework de Python para construir aplicaciones web.
CORS(app)  # Enable CORS for all routes and origins # Habilita el soporte de CORS (Cross-Origin Resource Sharing) para permitir el acceso a la API desde diferentes dominios y rutas.

api = Api(
    app,
    version='1.0',
    title='Car Price Prediction API',
    description='Car Price Prediction API')

ns = api.namespace('predict',
                   description='Car Price Predictor')

car = api.model('Car', { # Define un modelo de datos llamado "Car" utilizando el decorador api.model()
    'Year': fields.Integer(required=True, description='Car manufacture year'),
    'Mileage': fields.Integer(required=True, description='Car mileage'),
    'State': fields.String(required=True, description='Car registration state'),
    'Make': fields.String(required=True, description='Car make'),
    'Model': fields.String(required=True, description='Car model')
}) # El modelo define los campos requeridos para la entrada de datos de predicción de precios de automóviles


price_fields = api.model('Price', { # Define otro modelo de datos llamado "Price" que representa el campo "predicted_price" (precio predicho).
    'predicted_price': fields.Float
})


@ns.route('/')
class CarPriceApi(Resource): # Define una clase llamada "CarPriceApi" que hereda de la clase Resource de Flask-RESTful

    @api.expect(car)
    @api.marshal_with(price_fields)
    def post(self): # El método post() dentro de la clase CarPriceApi se ejecuta cuando se realiza una solicitud HTTP POST al endpoint "/predict/"
        data = api.payload

        year = data['Year']
        mileage = data['Mileage']
        state = data['State']
        make = data['Make']
        model = data['Model']  # Extrae los datos de entrada enviados en la solicitud, como el año, kilometraje, estado, marca y modelo del automóvil.

        # Crear un DataFrame con los datos ingresados
        input_data = pd.DataFrame([[year, mileage, state, make, model]], columns=["Year", "Mileage", "State", "Make", "Model"])

        # Utilizar el preprocesador y el modelo para predecir el precio del automóvil
        #Utiliza el modelo cargado para hacer predicción del precio 
        price_prediction = pipeline.predict(input_data)[0]

        return {"predicted_price": price_prediction}, 200 # Retorna un diccionario JSON que contiene el precio predicho del automóvil en el campo "predicted_price".



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
