import json
from openai import OpenAI
import numpy as np
from tqdm import tqdm
import re
import time
import pandas as pd



with open('dataset_ko.json', 'r') as file:
    data = json.load(file)

all_entities = data

def generate(prompts, n=1, max_tokens=512):
    key = "openai api key"
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model="gpt-4o", # gpt model
        n = n,
        max_tokens = max_tokens,
        messages=[{"role": "user", "content": prompts}])

    result = completion.choices[0].message.content
    texts = result.strip().split('\n')
    return texts



def sample_entities(all_entities, min_k=1, max_k=3):
    k = np.random.randint(min_k, max_k+1)
    idxs = np.random.choice(range(len(all_entities)), size=k, replace=False)

    entities = []
    for i in idxs:
        ents = all_entities[i]
        name = np.random.choice(ents['entity_names'])
        entities.append({'class_name': ents['class_name'], 'entity_name': name})
    
    return entities


def construct_sentence_prompt(entities, style='plane text'):
    prompt = f'Generate a {style} korean sentence including following entities.\n\n'

    entities_string = ', '.join([f"{e['entity_name']}({e['class_name']})" for e in entities])
    prompt += f'Entities: {entities_string}\n'
    prompt += 'Sentence:'
    return prompt


def construct_labels(generated, entities, class2idx):
    labels = [class2idx['outside']] * len(generated)
    for ent in entities:
        l = class2idx[ent['class_name']]
        for span in re.finditer(ent['entity_name'].lower(), generated.lower()):
            s, e = span.start(), span.end()
            labels[s] = l
            labels[s+1:e] = [l+1] * (e-s-1)
    return labels
    
   
class2idx = {e['class_name']: i*2 for i, e in enumerate(all_entities)}
class2idx['outside'] = len(class2idx) * 2

data = []
for _ in tqdm(range(100)):
    batch_entities = [sample_entities(all_entities) for _ in range(10)]
    batch_prompts = [construct_sentence_prompt(ents) for ents in batch_entities]

    for b,k in zip(batch_prompts, batch_entities):
        batch_generated = generate(b)
        labels = construct_labels(batch_generated[0], k, class2idx)
        data.append({'text': batch_generated[0], 'labels': labels})
    time.sleep(2)
        
    
    
print(data)

df = pd.DataFrame(data)
df.to_csv('20240514_1k_ko.csv', index=False)