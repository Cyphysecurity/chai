from openai import OpenAI, NOT_GIVEN
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
from PIL import Image
from transformers import pipeline
# from open_flamingo import create_model_and_transforms
import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
import rich
# from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline, BitsAndBytesConfig
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer
from util.internvl_image_loader import load_image_
def encode_image(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")

    image_bytes = buffered.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    # return encoded_image

    url = f"data:image/jpeg;base64,{encoded_image}"
    return url
class VLM():
    def __init__(self):
        pass
    def reset(self):
        raise NotImplementedError("reset method not implemented")
    def call_gpt(self, pmt, images, response_format=NOT_GIVEN):
        raise NotImplementedError("call_gpt method not implemented")
    
# class Llava(VLM):
#     def __init__(self):
#         super().__init__()
#         # quantization_config = BitsAndBytesConfig(
#         #     load_in_4bit=True,
#         #     bnb_4bit_compute_dtype=torch.float16,
#         # )
#         self.model = "llava-hf/llava-1.5-7b-hf"
#         self.client = pipeline("image-to-text", model=self.model)
#         self.temperature = 0.2
#         self.max_tokens = 2000
#     def reset(self):
#         # self.client = pipeline("image-text-to-text", model=self.model, device=0)
#         pass
#     def call_gpt(self, pmt, images, response_format=NOT_GIVEN):
#         """
#         pmt: str
#         images: list of PIL images
#         response_format: str
#         """
#         content=[
#                 {"type": "image_url", "image_url": {"url": encode_image( Image.fromarray(image))}}
#                 for image in images
#             ] + [{"type": "text", "text": pmt}]
#         return self._completion_retry(content)
#     def _completion_retry(self, content):
        
#         out = self.client(image, prompt=prompt, generate_kwargs={"max_new_tokens": 30})
#         return out

# class Qwen(VLM):
#     def __init__(self,model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
#         super().__init__()
#         self.model_name = model_name
#         self.model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             "Qwen/Qwen2.5-VL-7B-Instruct",
#             torch_dtype=torch.bfloat16,
#             attn_implementation="sdpa",
#             device_map="auto",
#         )
#         self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
#         # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.temperature = 0.2
#         self.max_tokens = 200
#     def reset(self):
#         self.model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             "Qwen/Qwen2.5-VL-7B-Instruct",
#             torch_dtype=torch.bfloat16,
#             attn_implementation="sdpa",
#             device_map="auto",
#         )
#         self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
#     def call_gpt(self, pmt, images, response_format=NOT_GIVEN):
#         # my_dict = {}
#         # for i in range(len(images)):
#         #     my_dict[f"image_{i}"] = images[i]
#         messages = [
#             {
#                 "role": "user", 
#                 "content": [{"type": "image", "image": encode_image(Image.fromarray(image))} for image in images
#                 ] + [{"type": "text", "text": pmt}],
#             }
#         ]
#         text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = inputs.to("cuda")
#         generated_ids = self.model.generate(**inputs, max_new_tokens=500)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = self.processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         # print(output_text)
#         return output_text[0].split("\n")[-1]
        
# class Flamingo(VLM):
#     def __init__(self):
#         super().__init__()
#         model, image_processor, tokenizer = create_model_and_transforms(
#         clip_vision_encoder_path="ViT-L-14",
#         clip_vision_encoder_pretrained="openai",
#         lang_encoder_path="anas-awadalla/mpt-7b",
#         tokenizer_path="anas-awadalla/mpt-7b",
#         cross_attn_every_n_layers=4
#         )
#         model.to(0)
#         self.client = model
#         self.image_processor = image_processor
#         self.tokenizer = tokenizer
#     def reset(self):
#         pass
#     def call_gpt(self, pmt, images, response_format=NOT_GIVEN):
#         """
#         pmt: str
#         images: list of PIL images
#         response_format: str
#         """
#         vision_x = [self.image_processor(Image.fromarray(demo_image_one)).unsqueeze(0) for demo_image_one in images]
#         vision_x = torch.cat(vision_x, dim=0).to(0)
#         vision_x = vision_x.unsqueeze(1).unsqueeze(0)
#         self.tokenizer.padding_side = "left"

        
#         lang_x = self.tokenizer(pmt, return_tensors="pt")
#         return self._completion_retry(vision_x, lang_x)
#     def _completion_retry(self, vision_x, lang_x):
        
#         generated_text = self.client.generate(vision_x.to(0), 
#                                               lang_x["input_ids"].to(0), 
#                                               attention_mask=lang_x["attention_mask"].to(0), 
#                                               max_new_tokens=500, 
#                                               temperature=0.2)
#         return self.tokenizer.decode(generated_text[0])

class InternVL(VLM):
    def __init__(self, gpu="cuda:0"):
        # path = "./models/InternVL2_5-8B"
        self.gpu = gpu
        path = "/home/cyphysecurity/Documents/llm/option_2/vehicle_LLM_attack/models/InternVL2_5-8B"
        self.model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().to(self.gpu)
        # self.image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px-V2_5')
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    def call_gpt(self, prompt, image, response_format = None, questions=None, question_2=None):
        image = [Image.fromarray(im) for im in image]
        pixel_values0 = load_image_(image[0]).to(torch.bfloat16).cuda(self.gpu)
        # pixel_values1 = load_image_(image[1]).to(torch.bfloat16).cuda(self.gpu)
        # pixel_values2 = load_image_(image[2]).to(torch.bfloat16).cuda(self.gpu)
        pixel_values3 = load_image_(image[3]).to(torch.bfloat16).cuda(self.gpu)
        # pixel_values4 = load_image_(image[4]).to(torch.bfloat16).cuda(self.gpu)
        # pixel_values5 = load_image_(image[5]).to(torch.bfloat16).cuda(self.gpu)
        # pixel_values = torch.cat((pixel_values0, pixel_values1, pixel_values2), dim=0) # TODO: improve this. This is only a test
        pixel_values = torch.cat((pixel_values0, pixel_values3), dim=0) # TODO: improve this. This is only a test
        num_patches_list = [pixel_values0.size(0), 
                            # pixel_values1.size(0), 
                            # pixel_values2.size(0), 
                            pixel_values3.size(0),
                            # pixel_values4.size(0), 
                            # pixel_values5.size(0),
                            ]
        # pv = (load_image_(image[i]) for i in range(len(image))) This is probably a better way of doing this
        # pixel_values = torch.cat(pv, dim=0) 
        # num_patches_list = [pv[i].size(0) for i in range(len(image))]
        generation_config = dict(max_new_tokens=1250, do_sample=True)
        # prompt = "Who are you?"
        response, history = self.model.chat(self.tokenizer, 
                                            pixel_values, 
                                            prompt, generation_config, 
                                            num_patches_list=num_patches_list, 
                                            history=None, 
                                            return_history=True)
        rp = response.split("Question")[-1]
        response_short, history = self.model.chat(self.tokenizer, 
                                            pixel_values, 
                                            "summarize the last answer in **ONE** word (proceed or brake):"  + rp, generation_config, 
                                            num_patches_list=num_patches_list, 
                                            history=history, 
                                            return_history=True)
        # for i in range(1, len(questions)):
        #     response, history = self.model.chat(self.tokenizer, 
        #                                         pixel_values, 
        #                                         questions[i], generation_config, 
        #                                         num_patches_list=num_patches_list, 
        #                                         history=history, 
        #                                         return_history=True)
        # print(f'Assistant: {response_short}')
        return response_short
    def reset(self):
        path = "/home/cyphysecurity/Documents/llm/option_2/vehicle_LLM_attack/models/InternVL2_5-8B"
        self.model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().to(self.gpu)
        # self.image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px-V2_5')
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    # def evaluate(self, out, flipped = False):
    #     return evaluator(out, flipped=flipped)


class GPT(VLM):
    def __init__(self, model="gpt-4o-2024-11-20", temperature=0.2, max_tokens=2000, api_key=""):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def reset(self):
        self.client = OpenAI(api_key=self.api_key)


    def call_gpt(self, pmt, images, response_format=NOT_GIVEN, questions=None, question_2=None):
        """
        pmt: str
        images: list of PIL images
        response_format: str
        """
        image_name = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        img = []
        content=[
                {"type": "image_url", "image_url": {"url": encode_image( Image.fromarray(image))}}
                for image in images
            ] + [{"type": "text", "text": pmt}]
        ct=[
            item
            for index, photo in enumerate(images)
            for item in (
                {"type": "text", "text": f"Image name: {image_name[index]}"},
                {"type": "image_url", "image_url": {"url": encode_image(Image.fromarray(photo))}}
            )
        ] + [{"type": "text", "text": pmt}]
        
        response = self._completion_retry(content, self.model, response_format)
        # print(response)
        if response_format == NOT_GIVEN:
            return response.choices[0].message.content
        else:
            # rich.print(response.choices[0].message.content)
            return json.loads(response.choices[0].message.content)["Answers"][-1]

    def _completion_retry(self,content, model, response_format=NOT_GIVEN):
        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format=response_format
        )
        if response.choices[0].finish_reason != "stop":
            raise ValueError(
                "Generation finish reason: {}. {}".format(
                    response.choices[0].finish_reason,
                    response.usage.to_dict()
                )
            )

        return response
    
