import requests
from loguru import logger
from PIL import Image

import torch
from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,

)

from huggingface_hub import hf_hub_download

class FlamengoWrapper(WrapperBase):

    def __init__(
        self,
        model_name="openflamingo/OpenFlamingo-9B-deprecated",
        enable_caching=True,
        simulate_time_delay=False,
        system_prompt=None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        # )
        # self.pipe = pipeline(
        #     "image-to-text",
        #     model=model_name,
        #     model_kwargs={"quantization_config": quantization_config},
        # )
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4
        )

        checkpoint = hf_hub_download(model_name, "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint), strict=False)
        self.model.to(self.device)



    def run_inference(self, prompt: str, image: Image):
        if not self.system_prompt:
            raise ValueError("System prompt is not set.")
        vision_x = [self.image_processor(image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        # prompt = "What is in the image?"
        lang_x = self.tokenizer(
            ['<image>'+prompt],
            return_tensors="pt",
            )
        """
        Step 4: Generate text
        """
        generated_text = self.model.generate(
            vision_x = vision_x.to(self.device),
            lang_x = lang_x["input_ids"].to(self.device),
            attention_mask = lang_x["attention_mask"].to(self.device),
            max_new_tokens=50,
            num_beams=3,
        )
        ans_formatted = self.tokenizer.decode(generated_text[0])
        # prompt_1_final = f"USER: <image>\n {self.system_prompt} ASSISTANT:"

        # ans = self.run_inference_inner(prompt_1_final, image)
        # answer_1 = ans.split("ASSISTANT:")[-1].strip()

        # prompt_2 = f"USER: <image>\n{self.system_prompt} ASSISTANT: {answer_1}</s>USER: {prompt} ASSISTANT:"
        # ans = self.run_inference_inner(prompt_2, image)
        # answer_2 = ans.split("ASSISTANT:")[-1].strip()

        # answer_2_yes_no = answer_2.lower().strip()
        # if "yes" in answer_2_yes_no:
        #     answer_2_yes_no = "yes"
        # elif "no" in answer_2_yes_no:
        #     answer_2_yes_no = "no"

        # ans_formatted = (
        #     f"Answer: {answer_2_yes_no} \nJustification: {answer_1}"
        # )

        return ans_formatted

    def run_inference_inner(self, prompt, image):
        # inputs = self.processor(text=prompt, images=image,
        #                        return_tensors = "pt")
        # inputs.to(self.device)

        # Generate
        # with torch.inference_mode():
        # generate_ids = self.model.generate(**inputs, max_new_tokens=30)

        #   ans = self.processor.batch_decode(
        #      generate_ids, skip_special_tokens=True,
        #     clean_up_tokenization_spaces=False)

        ans = self.pipe(
            image, prompt=prompt, generate_kwargs={"max_new_tokens": 30}
        )

        return ans[0]["generated_text"]


if __name__ == "__main__":

    system_prompt = "You are an intelligent AI assistant that helps an object detection system to identify objects of different classes in images."
    llava = FlamengoWrapper(system_prompt=system_prompt)

    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    # url = "https://www.scienceabc.com/wp-content/uploads/2018/09/injured-man.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    image = Image.open(
        "/home/cyphy-sec/Documents/drones/CloudTrack/assets/away-cars.png"
    )
    prompt = "Is there a police car of the Santa Cruz Police Department?"

    ans = llava.run_inference(prompt, image)
    print(ans)