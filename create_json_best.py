import numpy as np
import json
import pandas as pd

save = []
b_per_target = {}
scores = []
for target in ["a)", "b)", "c)"]:
    df = pd.read_csv(f"old_results/testing_False_training_data_True_universal_False_target_{target}.txt", sep=";")
    # print(df)
    dicti = []
    for i in range(14):
        data = df["image"] == i
        data = df[data]
        
        argm = data["success"].argmax()
        
        data = data[data["round"]==argm]
        
        rgb_letter = data["rgb_letter"]
        rgb_bck = data["rgb_bck"]
        text = data["optimal_txt"]
        text_id = data["optimal_txt_index"]
        if data["total_images"].iloc[0]==0:
            continue
        score = data["success"].iloc[0]/data["total_images"].iloc[0]*100
        if score > 50:
            scores.append(score)
        dicti.append({"rgb_bck":rgb_bck.iloc[0], "rgb_letter":rgb_letter.iloc[0], "text":text.iloc[0], "image_id":i, "text_id":int(text_id.iloc[0])})
    b_per_target[target] = dicti
    # save.append(b_per_target)
print(scores)
# print(np.mean(scores))
# print(np.std(scores))
with open("old_results/best_single.json", "w") as file:
    json.dump(b_per_target, file, indent=4)