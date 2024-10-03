import requests
from PIL import Image
from io import BytesIO
import time
import json
import jsonlines


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def log_time(message, start_time):
    end_time = time.time()
    print(f'{message} in {(end_time - start_time)} seconds')
    return end_time


def load_json(fn):
    with open(fn, encoding='utf8') as f:
        output = json.load(f)

    return output


def write_json(data, fn):
    with open(fn, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_jsonl(fn):
    output = []

    with open(fn, encoding='utf8') as f:
        list_str = list(f)
        for item_str in list_str:
            item_obj = json.loads(item_str)
            output.append(item_obj)

    return output


def write_jsonl(data, fn):
    fo = jsonlines.open(fn, 'w')
    for item in data:
        fo.write(item)
    fo.close()


def read_text(fn):
    with open(fn, encoding='utf8') as f:
        output = [l.strip() for l in f.readlines()]
    return output


def write_text(data, fn):
    with open(fn, 'w', encoding='utf8') as f:
        f.write('\n'.join(data))
