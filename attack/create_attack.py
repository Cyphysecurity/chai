import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO
from copy import deepcopy

class Attacker():
    def __init__(self):
        self.initial_position = (0, 0)
    
    def set_initial_position(self):
        pass

    def patch_image(self, image, attack, initial_position):
        size_attack = attack.shape
        size_image = image.shape

        size_x_f = min(size_image[0], initial_position[0]+size_attack[0])
        size_y_f = min(size_image[1], initial_position[1]+size_attack[1])
        image[initial_position[0]:size_x_f, initial_position[1]:size_y_f] = attack[0:size_x_f-initial_position[0], 0:size_y_f-initial_position[1]]
        return image
    def blend(self, x0, image, attack, alpha):
        size_attack = attack.shape
        size_image = image.shape
        image_copy = deepcopy(image)
        attack_copy = deepcopy(attack)

        for ch in range(3):
            for i in range(size_attack[0]):
                for j in range(size_attack[1]):
                    if alpha[i, j] > 0:
                        image_copy[x0[0]+i, x0[1]+j, ch] = attack[i, j, ch]
        return image_copy


    def attack(self, image, text, color_bck, color_txt, size, initial_position):
        image_attack = deepcopy(text)
        image_copy = deepcopy(image)
        
        for i in range(3):
            image_change = text[:, :, i] >100
            image_attack[image_change, i] = int(color_bck[i])
        for i in range(3):
            image_change = text[:, :, i] < 100
            image_attack[image_change, i] = int(color_txt[i])
        
        size_attack = image_attack.shape
        image_attack = cv2.resize(image_attack, (0, 0), fx=size, fy=size)
        # image_attack = cv2.cvtColor(image_attack, cv2.COLOR_BGR2RGB)
        return self.patch_image(image_copy, image_attack, initial_position)
    def _get_position(self, image, text, color_bck, color, scale, initial_position):
        text_sp = text.split(' ')
        im = deepcopy(image)
        _, bottom_left, text_width, text_height = self._put_text_en(im, text_sp[0], color_bck, color, scale, initial_position)
        pos2= (bottom_left[0], bottom_left[1]+text_height)
        im, bottom_left, text_width, text_height = self._put_text_en(im, text_sp[1], color_bck, color, scale, pos2)
        pos= (initial_position[0], initial_position[1] )
        im, bottom_left, _, text_height = self._put_text_en(im, text_sp[0], color_bck, color, scale, pos, text_width)
        return pos, pos2, text_width
    def put_text(self, image, text:str, color_bck, color, scale, initial_position, language="English"):
        if language == "Chinese":
            return self._put_text_ch(image, text, color_bck, color, scale, initial_position)
        else :
            
            if ' ' in text:
                text_sp = text.split(' ')
                pos, pos2, text_width = self._get_position(image, text, color_bck, color, scale, initial_position)
                image, _, _, _ = self._put_text_en(image, text_sp[0], color_bck, color, scale, pos, text_width)
                image, _, _, _ = self._put_text_en(image, text_sp[1], color_bck, color, scale, pos2, 0)
            else:
                image, bottom_left, text_width, text_height = self._put_text_en(image, text, color_bck, color, scale, initial_position)

            return image
            
    
    def _put_text_ch(self, image, text:str, color_bck, color, scale, initial_position):
        image_copy = image
        font_path = './attack/fonts/simsun.ttc'
        font = ImageFont.truetype(font_path, int(scale*40))
        image_copy = Image.fromarray(image_copy)
        draw = ImageDraw.Draw(image_copy)
        bbox = draw.textbbox(initial_position, text, font=font)
        draw.rectangle(bbox, fill=color_bck)
        draw.text(initial_position, text, font=font, fill=color)
        return np.array(image_copy)
    
    def _put_text_en(self, image, text:str, color_bck, color, scale, initial_position, width=0):
        image_copy = image
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 2

        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_width, text_height = text_size
        if width>0:
            initial_position = (initial_position[0]-(text_width-width)//2, initial_position[1] )

        bottom_left = initial_position 
        bottom_left = (bottom_left[0], bottom_left[1] + text_height//2 + 10) # -10 for single_example
        top_right = (initial_position[0] + text_width, initial_position[1] - text_height - 10)

        # Draw the background rectangle
        cv2.rectangle(image_copy, bottom_left, top_right, color_bck, thickness=cv2.FILLED)
        # Draw the text on top of the background
        text_position = (initial_position[0], initial_position[1] - 5)  # Adjust for padding
        cv2.putText(image_copy, text, text_position, font, scale, color, thickness)
        return image_copy, bottom_left, text_width, text_height

    def attack_distance(self, image1, image2):
        # Compute the distance between two images
        return np.linalg.norm(image1 - image2)
         
    
        
