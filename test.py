import json

with open('JSON_format.json') as f:
    JSON_templete = json.load(f)
print(JSON_templete)