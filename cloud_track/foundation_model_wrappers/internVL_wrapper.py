import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline,
    AutoTokenizer,
    AutoModel
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def preprocess_image(image, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternVLWrapper(WrapperBase):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL2_5-8B",
        enable_caching=True,
        simulate_time_delay=False,
        system_prompt=None,
        device="cpu",
    ):
        self.device = device
        path = "/home/cyphysecurity/Documents/llm/option_2/vehicle_LLM_attack/models/InternVL2_5-8B"
        self.model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        # )
        # self.pipe = pipeline(
        #     "image-to-text",
        #     model=model_name,
        #     model_kwargs={"quantization_config": quantization_config},
        # )

        # ###############
        # self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # self.system_prompt = system_prompt

    def run_inference(self, prompt: str, image: Image):
        
        

        prompt_1_final = f"<image>\n {prompt}"

        ans, history = self.run_inference_inner(prompt_1_final, image)
        answer_1 = ans.split("ASSISTANT:")[-1].strip()

        # prompt_2 = f"Summarize answer in one word, yes or no:"
        # ans, history = self.run_inference_inner(prompt_2, image, history=history)
        # answer_2 = ans.split("ASSISTANT:")[-1].strip()
        answer_2 = answer_1
        answer_2_yes_no = answer_2.lower().strip()
        if "yes" in answer_2_yes_no:
            answer_2_yes_no = "yes"
        elif "no" in answer_2_yes_no:
            answer_2_yes_no = "no"

        ans_formatted = (
            f"Answer: {answer_2_yes_no} \nJustification: {answer_1}"
        )

        return ans_formatted

    def run_inference_inner(self, prompt, image, history=None):
        # inputs = self.processor(text=prompt, images=image,
        #                        return_tensors = "pt")
        # inputs.to(self.device)

        # Generate
        # with torch.inference_mode():
        # generate_ids = self.model.generate(**inputs, max_new_tokens=30)

        #   ans = self.processor.batch_decode(  
        #      generate_ids, skip_special_tokens=True,
        #     clean_up_tokenization_spaces=False)
        generation_config = dict(max_new_tokens=60, do_sample=True)
        image = preprocess_image(image).to(torch.bfloat16).cuda(self.device)
        # prompt = '<image>\n In one word: Is there a police car of the Santa Cruz Police Department? '
        response, history = self.model.chat(self.tokenizer, image, prompt, generation_config, history=history, return_history=True)
        # print(history)
        # ans = self.pipe(
        #     image, prompt=prompt, generate_kwargs={"max_new_tokens": 50}
        # )

        return response, history #ans[0]["generated_text"]


if __name__ == "__main__":
    system_prompt = "You are an intelligent assistant, helping a drone on a search and rescue mission. Describe the image."
    llava = LlavaWrapper(system_prompt=system_prompt)

    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    # url = "https://www.scienceabc.com/wp-content/uploads/2018/09/injured-man.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    image = Image.open(
        "/home/blei/cloud_track/cloud_track/foundation_model_wrappers/images/cropped_image.png"
    )
    prompt = "Based on this information, would you recommend sending help? Answer with yes or no."

    ans = llava.run_inference(prompt, image)
    print(ans)
