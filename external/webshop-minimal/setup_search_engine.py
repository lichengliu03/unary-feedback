import sys
import json
from tqdm import tqdm
sys.path.insert(0, '../')

from utils import get_file_path
from engine import load_products

all_products, *_ = load_products(filepath=get_file_path())

import os
os.makedirs('./search_engine', exist_ok=True)
os.makedirs('./search_engine/resources_100', exist_ok=True)
os.makedirs('./search_engine/resources', exist_ok=True)
os.makedirs('./search_engine/resources_1k', exist_ok=True)
os.makedirs('./search_engine/resources_100k', exist_ok=True)

docs = []
for p in tqdm(all_products, total=len(all_products)):
    option_texts = []
    options = p.get('options', {})
    for option_name, option_contents in options.items():
        option_contents_text = ', '.join(option_contents)
        option_texts.append(f'{option_name}: {option_contents_text}')
    option_text = ', and '.join(option_texts)

    doc = dict()
    doc['id'] = p['asin']
    doc['contents'] = ' '.join([
        p['Title'],
        p['Description'],
        p['BulletPoints'][0],
        option_text,
    ]).lower()
    doc['product'] = p
    docs.append(doc)


with open('./search_engine/resources_100/documents.jsonl', 'w+') as f:
    for doc in docs[:100]:
        f.write(json.dumps(doc) + '\n')

with open('./search_engine/resources/documents.jsonl', 'w+') as f:
    for doc in docs:
        f.write(json.dumps(doc) + '\n')

with open('./search_engine/resources_1k/documents.jsonl', 'w+') as f:
    for doc in docs[:1000]:
        f.write(json.dumps(doc) + '\n')

with open('./search_engine/resources_100k/documents.jsonl', 'w+') as f:
    for doc in docs[:100000]:
        f.write(json.dumps(doc) + '\n')