from flask import Flask, render_template, request, jsonify, send_file

import requests
import json
from fastapi import FastAPI, Request
from fastapi.middleware.wsgi import WSGIMiddleware
import pandas as pd

app = Flask(__name__, template_folder='client', static_folder='client')


fastapi = FastAPI()

API_URL = "http://127.0.0.1:8000"

fuels = ['Diesel', 'Petrol', 'CNG', 'LPG']
owners = ['Un', 'Deux', 'Trois et plus']

@app.route('/')
def index():
    return render_template('index.html', Pred = '0.00')  # Remplacez nom

@app.route('/prediction', methods=['POST'])
def predict():
    json_data = request.get_json()
    transmission = json_data.get('transmission', '')
    fuel = json_data.get('fuel', '')
    owner = json_data.get('owner', '')
    year = int(json_data.get('year', 0))
    km_driven = float(json_data.get('km_driven', 0))
    engine = float(json_data.get('engine', 0))
    max_power = float(json_data.get('max_power', 0))

    transmission = 2 if transmission == 'Manual' else  1

    owner = owners.index(owner)
    
    fuel = fuels.index(fuel)+1

    data = {
        "transmission": transmission,
        "fuel": fuel,
        "owner": owner,
        "year": year,
        "km_driven": km_driven,
        "engine": engine,
        "max_power": max_power
    }

    # Send data to the FastAPI endpoint
    response = requests.post(f"{API_URL}/predict", json=data)

    if response.status_code == 200:
        result = response.json()
        print(result)
        return jsonify({"Predict": result['Predict']})
    else:
        return jsonify({"error": "Unable to get a prediction"})

        
@app.route('/predictionCSV', methods=['POST'])
def predictCSV():
        file = request.files['filecsv'] 
        # Extracting uploaded file name
        filename = file.filename
        # Save the file locally (you can customize this based on your requirements)
        file.save(filename)
        files = {'file': open(filename, 'rb')}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
        # Save the received content to a temporary file
            with open('downloaded_data.csv', 'wb') as file:
                file.write(response.content)

            # Read the CSV file using Pandas
            uploaded_df = pd.read_csv('downloaded_data.csv', encoding='unicode_escape')

            # Converting to HTML table
            uploaded_df_html = uploaded_df.to_dict(orient='records')

            # Render the HTML template with the data
            return render_template('index.html', data_var=uploaded_df_html)
        else:
            return "Failed to download CSV file."


@app.route('/download', methods=['GET'])
def download_file():
    # Specify the path to the file you want to make available for download
    file_path = 'downloaded_data.csv'
    
    # Specify the name you want the file to have when downloaded
    file_name = 'downloaded_data.csv'
    
    # Send the file to the client for download
    return send_file(file_path, as_attachment=True, download_name=file_name)
        

if __name__ == '__main__':
    fastapi.mount("/", WSGIMiddleware(app))
    app.run(debug=True, host='0.0.0.0', port=80)
