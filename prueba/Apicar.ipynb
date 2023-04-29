{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3b4015b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "pipeline = joblib.load(\"car_price_model.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abdca598",
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "654d806a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on all addresses (0.0.0.0)\n",
      " * Running on http://127.0.0.1:5000\n",
      " * Running on http://192.168.10.20:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [29/Apr/2023 13:30:57] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:30:57] \"GET /swaggerui/droid-sans.css HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:30:57] \"GET /swaggerui/swagger-ui.css HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:30:57] \"GET /swaggerui/swagger-ui-bundle.js HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:30:57] \"GET /swaggerui/swagger-ui-standalone-preset.js HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:30:57] \"GET /swagger.json HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:31:40] \"POST /predict/ HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:33:18] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:33:18] \"GET /swaggerui/droid-sans.css HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:33:18] \"GET /swaggerui/swagger-ui.css HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:33:18] \"GET /swaggerui/swagger-ui-bundle.js HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:33:18] \"GET /swaggerui/swagger-ui-standalone-preset.js HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:33:18] \"GET /swagger.json HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:36:39] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:36:39] \"GET /swaggerui/droid-sans.css HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:36:39] \"GET /swaggerui/swagger-ui.css HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:36:39] \"GET /swaggerui/swagger-ui-bundle.js HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:36:39] \"GET /swaggerui/swagger-ui-standalone-preset.js HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:36:39] \"GET /swagger.json HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [29/Apr/2023 13:36:39] \"GET /swaggerui/favicon-32x32.png HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask\n",
    "from flask_restx import Api, Resource, fields\n",
    "from flask_cors import CORS\n",
    "import pandas as pd\n",
    "\n",
    "app = Flask(__name__)\n",
    "CORS(app)  # Enable CORS for all routes and origins\n",
    "\n",
    "api = Api(\n",
    "    app,\n",
    "    version='1.0',\n",
    "    title='Car Price Prediction API',\n",
    "    description='Car Price Prediction API')\n",
    "\n",
    "ns = api.namespace('predict',\n",
    "                   description='Car Price Predictor')\n",
    "\n",
    "car = api.model('Car', {\n",
    "    'Year': fields.Integer(required=True, description='Car manufacture year'),\n",
    "    'Mileage': fields.Integer(required=True, description='Car mileage'),\n",
    "    'State': fields.String(required=True, description='Car registration state'),\n",
    "    'Make': fields.String(required=True, description='Car make'),\n",
    "    'Model': fields.String(required=True, description='Car model')\n",
    "})\n",
    "\n",
    "price_fields = api.model('Price', {\n",
    "    'predicted_price': fields.Float\n",
    "})\n",
    "\n",
    "\n",
    "@ns.route('/')\n",
    "class CarPriceApi(Resource):\n",
    "\n",
    "    @api.expect(car)\n",
    "    @api.marshal_with(price_fields)\n",
    "    def post(self):\n",
    "        data = api.payload\n",
    "\n",
    "        year = data['Year']\n",
    "        mileage = data['Mileage']\n",
    "        state = data['State']\n",
    "        make = data['Make']\n",
    "        model = data['Model']\n",
    "\n",
    "        # Crear un DataFrame con los datos ingresados\n",
    "        input_data = pd.DataFrame([[year, mileage, state, make, model]], columns=[\"Year\", \"Mileage\", \"State\", \"Make\", \"Model\"])\n",
    "\n",
    "        # Utilizar el preprocesador y el modelo para predecir el precio del automóvil\n",
    "        price_prediction = pipeline.predict(input_data)[0]\n",
    "\n",
    "        return {\"predicted_price\": price_prediction}, 200\n",
    "\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "572a308c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
