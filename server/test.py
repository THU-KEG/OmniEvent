import os
import requests 


url = "http://127.0.0.1:9621/query"
params = {
    "text": "U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn",
    "task": "Event Detection",
    # "triggers": [("assault", 113, 120)]
}

result = requests.post(url, json=params)
print(result.text)