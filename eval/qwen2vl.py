import base64
from argparse import ArgumentParser
import os
import json

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from utils import load_json


def load_image(image_path):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    decoded_image_text = encoded_image.decode('utf-8')
    base64_data = f"data:image;base64,{decoded_image_text}"
    return base64_data


def save_json(json_list, save_path):
    with open(save_path, 'w', encoding='utf8') as file:
        json.dump(json_list, file, indent=4, ensure_ascii=False)

class VQADataset(Dataset):
    def __init__(self, args, test):
        self.test = test
        self.args = args
        self.device = "cuda"
        self.build_model_and_tokenizer(args)
        self.build_processor(args)

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]
        image_path, question = data['image_path'], data['question']

        image = load_image(image_path)

        return dict(
            image=image,
            text=question
        )

    def build_processor(self, args):
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )


    def build_model_and_tokenizer(self, args):
        checkpoint = args.model_path
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device
        )

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def collate_fn(self, batch):
        messages = []
        for item in batch:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": item['image'],
                        },
                        {
                            "type": "text",
                            "text": item['text'] + '. Trả lời bằng một từ hoặc cụm từ.'
                        },
                    ],
                }
            ])

        # Prepare prompt
        tokenized_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=tokenized_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs


    def build_dataloader(self, args):
        dataloader = DataLoader(
            dataset=self,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=self.collate_fn,
        )

        return dataloader


    def eval(self, args, vqa_dataset, dataloader):
        for idx_batch, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = batch.to('cuda')

            # Generation configs
            generation_config = vqa_dataset.model.generation_config
            generation_config.do_sample = True
            generation_config.temperature = 0.2
            generation_config.top_k = 1
            generation_config.top_p = 0.001
            generation_config.max_new_tokens = 32
            generation_config.repetition_penalty = 1.1

            generated_ids = vqa_dataset.model.generate(
                **inputs, generation_config=generation_config
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for idx_response, response in enumerate(responses):
                self.test[idx_batch * args.batch_size + idx_response]['answer'] = response
        print(self.test[0])
        return self.test


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--bench_file", type=str, default="test.json")
    parser.add_argument(
        "--model_path",
        type=str,
        default="erax/EraX-VL-7B-V1"
    )
    parser.add_argument("--save_name", type=str, default="erax_vl")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Load test data
    data_path = args.bench_file
    data = load_json(data_path)

    # Load dataset
    vqa_dataset = VQADataset(args, data)

    # Build dataloader
    dataloader = vqa_dataset.build_dataloader(args)

    # Eval test data
    data = vqa_dataset.eval(args, vqa_dataset, dataloader)

    save_json(data, os.path.join(args.output_folder, f"{args.save_name}.json"))