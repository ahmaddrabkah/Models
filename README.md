# Models
This repository contains Machine learning(ML) code implemented using TensorFlow platform for three diseases COVID-19, Heart Stroke and Diabetes.
Also it contains the model loader which is an RestAPI implemented using Flask framework, this RestAPI is used as interface between the ML and Web app
(Which implemented using Spring Freamwork in Java).

# Installation 
 - First you need to install Python from: https://www.python.org/downloads/release/python-392/ 
   
 - Then Install Microsoft Visual C++ Redistributable : https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 
   (for Windows you need Make sure long paths are enabled https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing).
    
  **Run following commands while you are inside project directory** 
 - The create and start virtual environment 
    ```bash 
    python -m venv venv
    .venv/bin/activate
    ```
 - Then install the required package
    ```bash
    pip install -r requirements.txt
    ```
    
 # Run 
 - You can run the code inside [Creating_models_code](https://github.com/ahmaddrabkah/Models/tree/master/Creating_models_code) to create and train any model 
   for any of the brovided datasets by run these commands:
   ```bash
   .venv\Scripts\activate
   python (script name)
   ```
  - You can run model loader and use the RestAPI by run these commands :
   ```bash
   .venv\Scripts\activate
   python model_loader.py
   ```
   Then Send HTTP request with POST method and in the request body you need to pass model name(from the Models directory) and and the data you want to diagnose 
   as JSON object 
   example: {"modelName":"diabetes_model","data":[3,126,88,41,235,39.3,0.704,27]} you will get "NO" mean the result is negative.
   
# Online RestAPI 
This project is deployed into Microsoft Azure so you could access it online by this link : https://diagnoseme-models.azurewebsites.net 
   
