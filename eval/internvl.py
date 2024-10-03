from argparse import ArgumentParser
import os
import json

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from utils import load_json

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


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# https://huggingface.co/OpenGVLab/InternVL2-1B

def split_list(lst, n):
    length = len(lst)
    avg = length // n  # 每份的大小
    result = []  # 存储分割后的子列表
    for i in range(n - 1):
        result.append(lst[i * avg:(i + 1) * avg])
    result.append(lst[(n - 1) * avg:])
    return result


def save_json(json_list, save_path):
    with open(save_path, 'w', encoding='utf8') as file:
        json.dump(json_list, file, indent=4, ensure_ascii=False)

class VQADataset(Dataset):
    def __init__(self, test, input_size=448, use_thumbnail=True, dynamic_image_size=False, max_num=12):
        self.test = test
        self.input_size = input_size
        self.use_thumbnail = use_thumbnail
        self.dynamic_image_size = dynamic_image_size
        self.max_num = max_num
        self.transform = build_transform(input_size=input_size)

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]
        image, question = data['image_path'], data['question']

        # Add prompt same as VLMEvalKit
        # question = question + '. Trả lời bằng một từ hoặc cụm từ.'

        pixel_values = load_image(image)
        return {
            'question': question,
            'pixel_values': pixel_values,
        }


def collate_fn(batches):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    num_patches_list = [_['pixel_values'].size(0) for _ in batches]
    questions = [_['question'] for _ in batches]

    return pixel_values, questions, num_patches_list


def build_dataloader(args, data):
    dataset = VQADataset(test=data)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    return dataloader


def build_model_and_tokenizer(args):
    checkpoint = args.model_path
    model = AutoModel.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval()

    if args.vision_weight:
        model_contain_vision_weight = AutoModel.from_pretrained(
            args.vision_weight,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.vision_model = model_contain_vision_weight.vision_model
        del model_contain_vision_weight
    model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)

    return model, tokenizer


def eval(dataloader, model, tokenizer, data, args):
    for idx_batch, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        pixel_values = batch[0].to(dtype=torch.bfloat16, device='cuda')
        questions = batch[1]
        num_patches_list = batch[2]
        generation_config = dict(
            max_new_tokens=64,
            do_sample=False,
        )
        responses = model.batch_chat(tokenizer, pixel_values, questions, generation_config, num_patches_list=num_patches_list)

        for idx_response, response in enumerate(responses):
            data[idx_batch * args.batch_size + idx_response]['answer'] = response
    return data


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--bench_file", type=str, default="test.json")
    parser.add_argument(
        "--model_path",
        type=str,
        default="5CD-AI/Vintern-1B-v2"
    )
    parser.add_argument(
        "--vision_weight",
        type=str,
        default=None
    )
    parser.add_argument("--save_name", type=str, default="intern_t5")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Load test data
    data_path = args.bench_file
    data = load_json(data_path)

    # Build dataloader
    dataloader = build_dataloader(args, data)

    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(args)

    # Eval test data
    data = eval(dataloader, model, tokenizer, data, args)

    save_json(data, os.path.join(args.output_folder, f"{args.save_name}.json"))