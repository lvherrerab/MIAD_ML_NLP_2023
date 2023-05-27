from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import joblib
import os


app = Flask(__name__) # Importa y crea una instancia de la clase Flask para crear una aplicación web. Flask es un framework de Python para construir aplicaciones web.
CORS(app)  # Enable CORS for all routes and origins # Habilita el soporte de CORS (Cross-Origin Resource Sharing) para permitir el acceso a la API desde diferentes dominios y rutas.

api = Api(
    app,
    version='1.0',
    title='Movie Genre Prediction API',
    description='Movie Genre Prediction API')

ns = api.namespace('predict',
                   description='Movie Genre Predictor')

Texto = api.model('Texto', { # Define un modelo de datos llamado "Car" utilizando el decorador api.model()
    'Texto': fields.String(required=True, description='Ingresa el resumen de la trama de la pelìcula'),
    
    
}) # El modelo define los campos requeridos para la entrada de datos de predicción de precios de automóviles


movie_fields = api.model('MovieFields', {
    'genre_probabilities': fields.Raw,
})


# Cargar el modelo y el vectorizador



@ns.route('/')
class MovieApi(Resource): # Define una clase llamada "CarPriceApi" que hereda de la clase Resource de Flask-RESTful

    @api.expect(Texto)
    @api.marshal_with(movie_fields)
    def post(self): # El método post() dentro de la clase CarPriceApi se ejecuta cuando se realiza una solicitud HTTP POST al endpoint "/predict/"
        data = api.payload
        plot = data['Texto'] # Extrae los datos de entrada enviados en la solicitud, como el año, kilometraje, estado, marca y modelo del automóvil.
        
        # Load the model
        modelo = joblib.load(os.path.abspath("/Users/LinaH/Documents/Maestria/Machine_learning_PN/Week_7/Competencia/Api/peliculas.pkl")) 
        vectorizer= joblib.load(os.path.abspath("/Users/LinaH/Documents/Maestria/Machine_learning_PN/Week_7/Competencia/Api/vectorizer.pkl")) 
      
        
        #Procesamiento de texto
        # Procesamiento de texto utilizando el vectorizador cargado
        Texto_ing = vectorizer.transform([plot])
        
        
        #Make prediction
        
        predicted_probabilities = modelo.predict_proba(Texto_ing)
        genre_probabilities = predicted_probabilities[0].tolist()

        cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

        result = {col: probability for col, probability in zip(cols, genre_probabilities)}
        return {"genre_probabilities": result}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
