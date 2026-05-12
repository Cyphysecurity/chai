import argparse
import multiprocessing.pool
import scenic
from drivelm import DriveLM, DatasetLoad, ResponseFormat
from gpt.gpt_call import GPT, InternVL#, Llava, Flamingo
import cv2
from attack.create_attack import Attacker
from PIL import Image
import json
from util.misc_attack import DatabaseLoadCustom
# from augmentator.rain.effect_add import RainEffect
import copy
import multiprocessing
# import imgaug.augmenters as iaa
import numpy as np
import time
from util.data_saver import DataSaver
# multiprocessing.set_start_method('spawn')
from multiprocessing import Pool
import traceback

def get_scenic_scenario(sizes, targets, changing):
    """
    Function to create the scenic file to optimize the attack
    :param sizes: sizes of the attack dictionary
    :return: scenic scenario
    """
    sampler = "Verifai" if not changing else ""
    return scenic.scenarioFromString(f"""
param verifaiSamplerType = 'ce'
                                     

                                     
param size = {sampler}Range(0.8, 1.2)

param r_background = {sampler}DiscreteRange(0, 255)
param g_background = {sampler}DiscreteRange(0, 255)                                     
param b_background = {sampler}DiscreteRange(0, 255)
                                     
param r_letter = {sampler}DiscreteRange(0, 255)
param g_letter = {sampler}DiscreteRange(0, 255)
param b_letter = {sampler}DiscreteRange(0, 255)

# Optimal attack text                                
param optimal_1 = {sampler}DiscreteRange(1, {sizes[0]-1 if "a)" in targets else 0})
param optimal_2 = {sampler}DiscreteRange(0, {sizes[1]-1 if "b)" in targets else 0})
param optimal_3 = {sampler}DiscreteRange(0, {sizes[2]-1 if "c)" in targets else 0})
param optimal_4 = {sampler}DiscreteRange(0, {sizes[3]-1 if "d)" in targets else 0})
                                     
                                     
param pos_x = {sampler}DiscreteRange(-25, 25) 
param pos_y = {sampler}DiscreteRange(-25, 25) 
""")

def get_score(attack_params, initial_params, attacker:Attacker, images, agent:DriveLM, person=None, alpha=None, target=None, language="English", add_effect=False, vlm_name="gpt", name_answer=""):
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
    if target is None:
        raise
    # obtain attack parameters
    x               = initial_params["position"]["pos_x"] - attack_params["pos_x"]
    y               = initial_params["position"]["pos_y"] - attack_params["pos_y"]
    mult            = attack_params["size"]
    rgb_background  = attack_params["rgb_background"]
    rgb_letter      = attack_params["rgb_letter"]
    text            = attack_params["text"]
    size            = initial_params["size"]*mult
    initial_position = (y, x)
    
    image = copy.copy(images[0])
    if person is not None:
        image = attacker.blend((x-30,y+80), images[0], person, alpha)#, size)

    # deploy attack
    rgb_background = (255, 0, int(rgb_background[2]))
    rgb_letter = (int(rgb_letter[0]), int(rgb_letter[1]), int(rgb_letter[2]))
    image = attacker.put_text(image, text.upper(), rgb_background, rgb_letter, size, initial_position, language=language)
    
    if attack_bool:
        images[0] = image
    Image.fromarray(images[0]).save("delete_me.png", quality=100)

    # Call agent to get the answer
    answer = agent.get_answer(images, response_format=ResponseFormat)
    
    # Calculate score
    if target in answer or name_answer[target] in answer.lower():
        score = -1
    else:
        score = 1
    if testing:
        pass
    return score, name_answer[target]

def deploy_attack(vlm_name, data, images, effect, attack_params, target, language, name_answer, gpus=0):
    if vlm_name == "internvl":
        vlm = InternVL(gpus)
    else:
        vlm = GPT()
    agent = DriveLM(vlm)
    agent.init_prompt()
    attacker = Attacker()
    alpha = person = None
    initial_params = {"position": data["position"], "size": data["size"]}
    
    res = 0
    image = 0
    # Try to get the score, if fails reset the vlm and try again
    try:
        res, _ = get_score(attack_params, initial_params, attacker, images, agent, target=target, language=language, add_effect=effect, person=person, alpha=alpha, vlm_name=vlm_name, name_answer=name_answer)
    except:
        try:
            time.sleep(5)
            vlm.reset()
            res, _ = get_score(attack_params, initial_params, attacker, images, agent, target=target, language=language, add_effect=effect, person=person, alpha=alpha, vlm_name=vlm_name, name_answer=name_answer)
        except Exception as e:
            print("Error: ", data["target"])
            traceback.print_exc() # Prints the full traceback to standard error
            print(f"Error message: {e}") 
    
    return res, target

def load_universal(vlm_name):
    '''
    Function to load the universal attack parameters
    :param vlm_name: name of the vlm
    :return: attack parameters
    '''
    if vlm_name == "gpt":
        with open('data.json', 'r') as file:
            data = json.load(file)
        r_background = data["r_background"]
        g_background = data["g_background"]
        b_background = data["b_background"]
        r_letter = data["r_letter"]
        g_letter = data["g_letter"]
        b_letter = data["b_letter"]
        
        optimal_text_select = [data["optimal_a"], data["optimal_b"], data["optimal_c"], data["optimal_d"]]
        size = 1.07
    elif vlm_name == "internvl":
        r_letter = 0
        g_letter = 0
        b_letter = 0
        r_background = 0
        g_background = 0
        b_background = 0
        
        optimal_text_select = [0, 0, 0, 0]
        size = 0.93
    return r_background, g_background, b_background, r_letter, g_letter, b_letter, optimal_text_select, size
def get_attack_params(params, target, texts, universal, i_image, vlm_name, changing):
    '''
    Function to get the attack parameters
    :param scene: scenic scene
    :param target: target action
    :param texts: dictionary of texts
    :param universal: whether to use universal attack or not
    :param i_image: index of the image
    :param vlm_name: name of the vlm
    :param changing: what parameter is changing
    :return: attack parameters'''
    targets = ["a)", "b)", "c)", "d)"]
    if vlm_name == "internvl":
        targets = ["a)", "b)"]
    # params = scene.params
    if testing:
        if universal:
            r_background, g_background, b_background, r_letter, g_letter, b_letter, optimal_text_select, size = load_universal(vlm_name)  
            pos_x = 0
            pos_y = 0
        else:
            raise("not yet implemented")      
    else:
        r_background = params["r_background"]
        g_background = params["g_background"]
        b_background = params["b_background"]
        r_letter = params["r_letter"]
        g_letter = params["g_letter"]
        b_letter = params["b_letter"]
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        optimal_text_select = [params["optimal_1"], params["optimal_2"], params["optimal_3"], params["optimal_3"]]
        if vlm_name == "internvl":
            optimal_text_select = [params["optimal_1"], params["optimal_2"]]
        try:
            size = params["size"]
        except:
            size = 1.07  
    rgb_background = (r_background, g_background, b_background)
    rgb_letter = (r_letter, g_letter, b_letter)
    counter = -1
    for key in texts:
        counter += 1
        if key["target"] == target:
            possible_texts = key["attacks"]
            break
    idx = optimal_text_select[counter]
    try:
        text = possible_texts[language][idx]
    except:
        print("Error: ", possible_texts, idx, counter)
        raise

    return {"size":size, "rgb_background": rgb_background, "rgb_letter": rgb_letter, "text":text, "text_id": idx, "pos_x": pos_x, "pos_y": pos_y}



def run_universal(vlm_name, dataset, vp_dict_size, iterations, targets, vp_texts, language, parallel, imgs, universal, changing=""):
    '''
    Function to run CHAI in DriveLM. This one is a bit more messy because it handles various target outputs.
    '''
    # ------------------
    # Setup saving file
    if vlm_name == "gpt":
        name_answer = {"a)": "proceed", "b)":"brake"}
    else:
        name_answer = {"a)": "proceed", "b)": "turn left", "c)": "turn right", "d)":"brake"}
    file_name = "universal" if universal else f"universal_{imgs[0]}"
    if len(targets)==1:
        file_name += f"_target_{name_answer[targets[0]]}"
    file_name = f"{file_name}_training_{not testing}_training_ds_{training_dataset}_vlm_name_{vlm_name}"
    file_name += f"_changing_{changing}" if changing != "" else ""
    file_name += f"_scenetap_{use_scenetap}"
    file_name += f"_attackbool_{attack_bool}_lastf_fpass.txt"
    
    data_saver = DataSaver(file_name, targets, name_answer)
    # ------------------
    # Setup scenario
    result = None
    scenario = get_scenic_scenario(vp_dict_size, targets, changing)
    for j in range(iterations): # maximum number of optimizer iterations
        scores_per_target = {}
        for target in targets:
            scores_per_target[target] = [0, 0]
        scene, _ = scenario.generate(feedback = result)
        params = copy.deepcopy(scene.params)
        result = 0
        succesfull = 0
        n_test_images = 0
        
        data_parallel = []
        optimal_vps = []
        counter = 0
        for target in targets: # Iterate over all possible targets
            effect = None
            result_per_target = 0
            for img_n in imgs: # Iterate over all the images
                images, data = dataset.get_item(img_n)

                # continue if this data is the same as the target
                if data[f"correct_{vlm_name}"] == target and not getting_baseline: 
                    continue

                # obtain attack parameters
                attack_params = get_attack_params(params, target, vp_texts, universal, img_n,vlm_name, changing)
                n_test_images += 1
                gpu = gpus[counter%len(gpus)]

                if parallel: # Create the attack data for parallel processing
                    data_parallel.append([vlm_name, data, images, effect, attack_params, target, language, name_answer, gpu])
                else: # Run one iteration sequentially
                    r, image = deploy_attack(vlm_name, data, images, effect, attack_params, target, language, name_answer, gpu)
                    result_per_target += r
                    result += r
                    succesfull -= r if r<0 else 0
                counter += 1
            try:
                optimal_vps.append(attack_params["text"])
            except:
                optimal_vps.append("None")
            if not parallel:
                scores_per_target[target][0] += result_per_target
                scores_per_target[target][1] += 1                
        
        if parallel:
            # Run the attacks in parallel
            if vlm_name == "gpt":
                r_img = []
                with Pool() as pool:
                    r_img = pool.starmap(deploy_attack, data_parallel)
            else:
                n = len(gpus)
                parallel = [data_parallel[i:i + n] for i in range(0, len(data_parallel), n)]
                r_img = []
                for p in parallel:
                    with Pool() as pool:
                        r_temp = pool.starmap(deploy_attack, p)
                    r_img.append(r_temp[0])
            r = []
            for res, target in r_img:
                r.append(res)
                scores_per_target[target][0] += res
                scores_per_target[target][1] += 1

            result = sum(r)
            succesfull = [res if res<0 else 0 for res in r]
            succesfull = -sum(succesfull)

        data_saver.add_data(attack_params, result, j, 0, succesfull, optimal_vps, n_test_images, targets, scores_per_target)
        data_saver.save()




    
def main():
    changing = ""
    if not(changing == ""):
        raise(NotImplemented)
    if testing and changing != "":
        raise("Should not run this expermient")
    
    if training_dataset:
        data_file = "training.json"
    else:
        data_file = "testing.json"
    
    dataset = DatabaseLoadCustom(data_file, llm_name=vlm_name, use_scenetap=use_scenetap)

    with open(f"attack_dicts_{vlm_name}.json", "r") as f:
        texts = json.load(f)
    sizes = []
    for vect_text in texts:
        sizes.append(len(vect_text["attacks"][language]))
    
    # if random:
    #     scenario = get_scenic_scenario_rand(sizes)

    if vlm_name == "gpt":
        targets = ["a)", "b)", "c)", "d)"]
        if use_scenetap:
            targets = ["a)", "d)"]
    else:
        targets = ["a)", "b)"]

    targets = ["a)"]    
    if getting_baseline:
        targets = [targets[0]]
  
    parallel = False
    
    iterations = 20 if testing else 150
    imgs = range(dataset.get_size())
    universal = True
    run_universal(vlm_name, dataset, sizes, iterations, targets, texts, language, parallel, imgs, universal, changing)


# These are the parameters that can be changed for the different experiments. They are defined here to avoid confusion and to make it easier to run the different experiments.
# TODO: make a better way to handle the different experiments and parameters, maybe with a config file or command line arguments
testing = True
training_dataset = True
attack_bool = True

getting_baseline = False
use_scenetap = False
attack_bool = attack_bool if not getting_baseline else False
attack_bool = attack_bool if not use_scenetap else False
testing = testing if not use_scenetap else True
language = "English"
gpus = ["cuda:0"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DriveLM experiments either full optimization")
    parser.add_argument("--testing", action="store_true", help="Run in testing mode (training otherwise)")
    parser.add_argument("--testing_ds", action="store_true", help="Run over testing dataset (training otherwise)")
    parser.add_argument("--vlm", type=str, default="gpt", help="Name of the VLM to attack (gpt or internvl)")
    args = parser.parse_args()
    if args.testing:
        testing = True
    if args.testing_ds:
        training_dataset = False
    if args.vlm:
        vlm_name = args.vlm
        assert vlm_name in ["gpt", "internvl"], "VLM name must be either gpt or internvl"
    main()
    

