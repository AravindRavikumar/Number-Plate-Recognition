import requests 
import json
import mimetypes
from pprint import pprint
    
with open('Resources/Indian_Number_Plates.json') as json_data:
    data = json.load(json_data)
i = 0
for d in data:
    image_url = d['content']
    try:
        response = requests.get(image_url) 
        content_type = response.headers['content-type']
        extension = mimetypes.guess_extension(content_type)
        with open(f"Indian_images/image{i}.{extension}",'wb') as f: 
            f.write(response.content) 
    except requests.exceptions.RequestException as e: 
        print(f"{i} not successful.")
    i = i+1
    print(i)

