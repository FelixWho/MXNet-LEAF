import json
import math

path = "/Users/shubeini/Downloads/all_data_0_niid_05_keep_0_train_9.json"

with open(path) as f:
    data = json.load(f)

print(data['user_data']['f0001_41']['x'][0])
