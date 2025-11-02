import requests
import pandas as pd
url="http://127.0.0.1:5000/linear_model_predict"
payload = {"my_input":[-0.0380759, -0.051474, 0.061696]}
response = requests.post(url,json=payload)
print(response.status_code)
print(response.json())