import cv2
from PIL import Image
import json
from openai import  NOT_GIVEN
import matplotlib.pyplot as plt
import numpy as np
from gpt.gpt_call import GPT#, Llava, Flamingo


class DatasetLoad():
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            data_all = json.load(f)
        self.data = data_all
    
    def get_size(self):
        return len(self.data)

    def get_item(self, idx):
        data_item = self.data[idx]
        filename = data_item['image']
        ids = data_item['id']
        question = data_item['conversations'][0]['value']
        answer = data_item['conversations'][1]['value']
        
        # prompt = llama.format_prompt(question)
        
        if isinstance(filename, list):
            image_all = []
            for img_path in filename:
                image_raw = cv2.imread(img_path)
                image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                image_all.append(image_raw)
        else:
            image = cv2.imread(filename)
            image = Image.fromarray(image)
            if self.transform:
                image = image

        return image_all

from typing import List
from pydantic import BaseModel
class ResponseFormat(BaseModel):
    Answers: List[str]

class DriveLM():
    def __init__(self, vlm:GPT):
        self.vlm = vlm
        self.pmt_2 = None
        
    
    def init_prompt(self):
        self.questions_pred = ["\nWhat are the important objects in the current scene?",
            '\nWhat is the moving status of each object?',
            '\nWhat is the visual description of each object?',
            '\nIs any object in the photos a signal?',
            '\nWhat is the future state of each object?',
            "\nAnalyze each object to determine if it's moving in the direction of the ego vehicle"
            '\nWhat object should the ego vehicle notice first / second / third when the ego vehicle is getting to the next possible location?',
            '\nWhat actions could the ego vehicle take based on the observation of each object?',
            '\nWhat actions taken by the ego vehicle can lead to a collision with each object?',
            '\nIn this scenario, what are safe actions to take for the ego vehicle?',
            "\nBased on previous questions, choose the action the vehicle should take among: a) proceed, b) turn left, c) turn right, d) brake."]
        
        if isinstance(self.vlm, GPT):
            pmt = 'You have 6 photos of a vehicle: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT. Answer the following questions: '
            pmt += "\n Be careful with camera names and be as succinct as possible. For multiple choice questions, please answer with the letter only and then parentheses, for example, a). \n"
        else:
            pmt = 'You have 2 photos of a vehicle: <image> CAM_FRONT, <image> CAM_BACK. Answer the following questions : \n'
            # pmt += "\n Becareful with camera names and be as succinct as possible. For multiple choice questions, please answer with the letter only and then parenthesis, for example, b). \n"
            self.questions_pred = ["\nWhat are the important objects in the current scene?",
            '\nWhat is the moving status of each object?',
            '\nWhat is the visual description of object?',
            '\nIs any object in the photos a signal or red light?',
            '\nWhat is the future state of each object?',
            "\nAnalyze each object to determine if it's moving in the direction of the ego vehicle"
            '\nWhat object should the ego vehicle notice first / second / third when the ego vehicle is getting to the next possible location?',
            '\nWhat actions could the ego vehicle take based on the observation of each object?',
            '\nWhat actions taken by the ego vehicle can lead to a collision with each object?',
            '\nIn this scenario, what are safe actions to take for the ego vehicle?',
            "\nBased on previous questions, choose the action the vehicle should take between: a) proceed, b) brake."
            ]
            # pmt = 'You have 6 photos of a vehicle: <image>CAM_FRONT, <image>CAM_FRONT_LEFT, <image>CAM_FRONT_RIGHT, <image>CAM_BACK, <image>CAM_BACK_LEFT, <image>CAM_BACK_RIGHT. Answer the following questions : '
        # pmt += "\n Becareful with camera names and be as succinct as possible. For multiple choice questions, please answer with the letter only and then parenthesis, for example, a). \n"
        # elif isinstance(self.vlm, Flamingo):
        #     pmt = 'You have 6 photos of a vehicle: <image>CAM_FRONT<|endofchunk|>, <image>CAM_FRONT_LEFT<|endofchunk|>, <image>CAM_FRONT_RIGHT<|endofchunk|>, <image>CAM_BACK<|endofchunk|>, <image>CAM_BACK_LEFT<|endofchunk|>, <image>CAM_BACK_RIGHT<|endofchunk|>. Answer the following questions : '
        # else:
        #     pmt = 'You have 6 photos of a vehicle: <image>CAM_FRONT<|im_end|>, <image>CAM_FRONT_LEFT<|im_end|>, <image>CAM_FRONT_RIGHT<|im_end|>, <image>CAM_BACK<|im_end|>, <image>CAM_BACK_LEFT<|im_end|>, <image>CAM_BACK_RIGHT<|im_end|>. Answer the following questions : '
        for i, prompt in enumerate(self.questions_pred):
            if not isinstance(self.vlm, GPT) and i==len(self.questions_pred)-1:
                # continue
                pass
            pmt += f"Question {i}: " + prompt + "\n"
        
        self.pmt = pmt
        if not isinstance(self.vlm, GPT):
            self.pmt_2 = self.questions_pred[-1]
    
    def get_answer(self, images, response_format=NOT_GIVEN):
        ans = self.vlm.call_gpt(self.pmt, images, response_format, self.questions_pred, self.pmt_2)
        ans = ans.lower()
        print(f'Assistant: {ans}')
        if not isinstance(self.vlm, GPT):
            if ("a)" not in ans and "proceed" not in ans) and ("b)" not in ans and "brake" not in ans):
                self.vlm.reset()
                return self.vlm.call_gpt(self.pmt, images, response_format, self.questions_pred, self.pmt_2)
                print(f'Assistant: {ans}')
        return ans
    
    def get_prompt(self):
        return self.pmt





if __name__ == '__main__':
    data_file = "test_llama.json"
    dataset = DatasetLoad(data_file)
    vlm = GPT()
    agent = DriveLM(vlm)
    agent.init_prompt()
    for i in range(10):
        images = dataset.get_item(0)
        answer = agent.get_answer(images, response_format=ResponseFormat)
        
        
    




