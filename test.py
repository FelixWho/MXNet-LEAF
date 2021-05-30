import json
filepath = "/home/fwh7/FederatedLearningFramework/leaf/data/sent140/data/train/all_data_niid_05_keep_3_train_9.json"

with open(filepath) as f:
    data = json.load(f)

for user in data['users']:
    user_x = []
    user_y = []
    for x in data['user_data'][user]['x']:
        user_x.append(x[4])

print(user_x)
                #    user_x = []
                #    user_y = []
                #    for x in data['user_data'][user]['x']:
                #        user_x.append(x)
                #    for y in data['user_data'][user]['y']:
                #        y = np.float32(y)
                #        user_y.append(y)
