import scenic
from drivelm import DriveLM, DatasetLoad, ResponseFormat
from gpt.gpt_call import GPT, Llava, Flamingo
import cv2
from attack.create_attack import Attacker
from PIL import Image
import json
from util.misc_attack import DatabaseLoadCustom
from augmentator.effect_add import RainEffect
import copy
import numpy as np
import torch



rain_adder = RainEffect()
def get_score(patch, initial_params, attacker:Attacker, images, texts:dict, agent:DriveLM, person=None, alpha=None, target=None, language="English", add_effect=False):
    """
    Function to get the score of the attack
    :param params: parameters of the attack that come from the scenic file
    :param initial_params: initial parameters of the attack
    :param attacker: attacker object
    :param images: images to attack
    :param texts: texts to attack
    :param agent: agent object (eg. DriveLM)
    :param person: person image
    :param alpha: alpha image
    :param target: target action"""
    # Get the initial position of the attack and add some bias based on optimizer
    x = (images[0].shape[1] - patch.shape[1] )//2
    y = (images[0].shape[0] - patch.shape[0] )//2
    initial_position = (y, x)

    image = copy.copy(images[0])
    # patch_ten = torch.from_numpy(patch)

    # image_tensor = torch.from_numpy(image)

    if person is not None:
        image = attacker.blend((x,y), images[0], person, alpha)#, size)

    
    # image = attacker.put_text(image, text, rgb_background, rgb_letter, size, initial_position, language=language)
    image = attacker.patch_image(image, patch.T, initial_position)
    distance = attacker.attack_distance(images[0], image)
    # Change image
    images[0] = image
    # Add rain effect
    if add_effect:
        for it, img in enumerate(images):
            image_rain = rain_adder.add_rain_effect(Image.fromarray(img))
            images[it] = image_rain
    

    
    answer = agent.get_answer(images, response_format=ResponseFormat)
    # answer = {"Answers": ["d)"]}
    if target in answer["Answers"][-1]:
        score = -1
    else:
        score = 1
    
    return score + 0*5*0.000001*distance, (params["pos_y"], params["pos_x"]), rgb_background, rgb_letter, params["size"], optimal_text_select, images[0] # No regularization to begin with
result = None

if __name__ == '__main__':
    language = "English"
    vlm_name = "flamingo"
    data_file = "attack.json"
    dataset = DatabaseLoadCustom(data_file)
    if vlm_name == "flamingo":
        vlm = Flamingo()
    else:
        vlm = GPT()
    agent = DriveLM(vlm)
    agent.init_prompt()
    attacker = Attacker()


    with open("attack_dicts.json", "r") as f:
        texts = json.load(f)
    sizes = []
    for vect_text in texts:
        sizes.append(len(vect_text["attacks"][language]))
    
    patch_names = {"a)": "patches/a) proceed_simple_patch_1000/final_best_patch.npy",
                   "b)": "patches/b) turn left_simple_patch_1000/final_best_patch.npy", 
                   "c)": "patches/c) turn right_simple_patch_1000/final_best_patch.npy",
                   "d)": "patches/d) brake_simple_patch_1000/final_best_patch.npy"}
    for j in range(100): # maximum number of optimizer iterations
        


        result = 0
        vlm.reset()
        effects = [False]

        n_test_images = 0
        for effect in effects:
            for i in range(dataset.get_size()): # Iterate over all images
                
                n_test_images += 1
                images, data = dataset.get_item(i)
                if data["target"] == "d)":
                    continue
                params = {"position": data["position"], "size": data["size"]}

                patch = np.load(patch_names[data["target"]])[0]
                patch = patch*255
                patch = patch.astype(np.uint8)
                res = 0
                try:
                    res, initial_position, rgb_background, rgb_letter, size, optimal_text_select, image = get_score(patch, params, attacker, images, texts, agent, target=data["target"], language=language, add_effect=effect)
                except:
                    try:
                        vlm.reset()
                        res, initial_position, rgb_background, rgb_letter, size, optimal_text_select, image = get_score(params, attacker, images, texts, agent, target=data["target"], language=language, add_effect=effect)
                    except:
                        print("Error: ", data["target"], i)

                result += res
                Image.fromarray(image).save(f"attacks/image_{i}_attack_{j}_lang_{language}_rainy_{effect}_exp.png")
            #  result += n_test_images
            #  result /= n_test_images*2
        text_print = f'Round {j}; {result}; pos_delta; {initial_position}; rgb_background; {rgb_background}; rgb_letter; {rgb_letter}; size; {round(size,2)}'
        for i, texts_att in enumerate(texts):
            if texts_att["target"] == "d)":
                continue
            txt = texts_att["attacks"][language]
            text_print += f'; optimal_{i}: {txt[optimal_text_select[i]]}'
        print(text_print)
        
        
        

    
    

