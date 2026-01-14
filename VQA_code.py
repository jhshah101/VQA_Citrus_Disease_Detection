# -*- coding: utf-8 -*-
"""Deep Learning-based Visual Question Answering System for Dental Diseases Identification.ipynb




# Visual Question Answering using Multimodal Transformer Models

## Import necessary libraries & set up the environment
"""

!pip install datasets==1.17.0

!pip install pandas==1.3.5

!pip install nltk==3.5

import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)

# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet

from sklearn.metrics import accuracy_score, f1_score

# from google.colab import drive
# drive.mount('/content/drive')

# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join("/kaggle/working", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_caching_enabled(True)
logging.set_verbosity_error()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

"""## Load the customized Dental Dataset

All the questions have 1-word/phrase answer, so we consider the entire vocabulary of answers available (*answer space*) & treat them as labels. This converts the visual question answering into a multi-class classification problem.
"""

dataset = load_dataset(
    "csv",
    data_files={
        "train": "/kaggle/input/vqa-thesis/yesno_dataset/data_train.csv",
        "test": "/kaggle/input/vqa-thesis/yesno_dataset/data_eval.csv"
    },
    # following 3 lines separate each column in train and test files
    # delimiter="\t", # Specify the delimiter as tab
    # header=None, # To avoid the assumption that the first line is a header and it becomes the only column name
    # column_names=['question', 'answer', 'image_id'] #Specify the column names for new ones
)


with open("/kaggle/input/vqa-thesis/answer_space.txt") as f:
    answer_space = f.read().splitlines()

dataset = dataset.map(
    lambda examples: {
        'label': [
            answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
                        # if ans.replace(" ", "").split(",")[0] in answer_space # Check if preprocessed answer is in answer_space
            # else -1  # or assign a default value if not found, e.g., -1
            if isinstance(ans, str) else
            answer_space.index(str(ans))  # Handle int answers
            for ans in examples['answer']
        ]
    },
    batched=True
)

dataset





"""### Look at some of the Question/Image/Answer combinations"""

from IPython.display import display

def showExample(train=True, id=None):
    if train:
        data = dataset["train"]
    else:
        data = dataset["test"]
    if id == None:
        id = np.random.randint(len(data))
    image = Image.open(os.path.join("/kaggle/input/vqa-thesis", "images", data[id]["image_id"] + ".jpg")).resize((350, 350))
    display(image)

    print("Question:\t", data[id]["question"])
    print("Answer:\t\t", data[id]["answer"], "(Label: {0})".format(data[id]["label"]))

showExample()

"""### Create a Multimodal Collator for the Dataset

This will be used in the `Trainer()` to automatically create the `Dataloader` from the dataset to pass inputs to the model

The collator will process the **question (text)** & the **image**, and return the **tokenized text (with attention masks)** along with the **featurized image** (basically, the **pixel values**). These will be fed into our multimodal transformer model for question answering.
"""

@dataclass
class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[Image.open(os.path.join("/kaggle/input/vqa-thesis", "images", image_id + ".jpg")).convert('RGB') for image_id in images],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }

    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }

"""## Defining the Multimodal VQA Model Architecture

Multimodal models can be of various forms to capture information from the text & image modalities, along with some cross-modal interaction as well.
Here, we explore **"Fusion" Models**, that fuse information from the text encoder & image encoder to perform the downstream task (visual question answering).

The text encoder can be a text-based transformer model (like BERT, RoBERTa, etc.) while the image encoder could be an image transformer (like ViT, Deit, BeIT, etc.). After passing the tokenized question through the text-based transformer & the image features through the image transformer, the outputs are concatenated & passed through a fully-connected network with an output having the same dimensions as the answer-space.

Since we model the VQA task as a multi-class classification, it is natural to use the *Cross-Entropy Loss* as the loss function.
"""

text_encoder = 'bert-base-uncased'
vision_encoder = 'google/vit-base-patch16-224-in21k'

class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_labels: int = len(answer_space),
            intermediate_dim: int = 1536,
            pretrained_text_name: str = text_encoder,
            pretrained_image_name: str = vision_encoder):

        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name

        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        self.fusion = nn.Sequential(
            #nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim), # in case of concatenation of features
            nn.Linear(self.text_encoder.config.hidden_size, intermediate_dim), # in case of mul or add of features
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Linear(intermediate_dim, self.num_labels)

        self.criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):

        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            # torch.cat(
            #     [
            #         encoded_text['pooler_output'],
            #         encoded_image['pooler_output'],
            #     ],
            #     dim=1
            # )
            torch.add(
               encoded_text['pooler_output'],
               encoded_image['pooler_output']
            )
        )
        logits = self.classifier(fused_output)

        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss

        return out

"""### Define a Function to Create the Multimodal VQA Models along with their Collators

We plan to experiment with multiple pretrained text & image encoders for our VQA Model. Thus, we will have to create the corresponding collators along with the model (tokenizers, featurizers & models need to be loaded from same pretrained checkpoints)
"""

def createMultimodalVQACollatorAndModel(text=text_encoder, image=vision_encoder):
    tokenizer = AutoTokenizer.from_pretrained(text)
    preprocessor = AutoFeatureExtractor.from_pretrained(image)

    multi_collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
    )


    multi_model = MultimodalVQAModel(pretrained_text_name=text, pretrained_image_name=image).to(device)
    return multi_collator, multi_model

"""## Performance Metrics from Visual Question Answering

### Wu and Palmer Similarity

The Wu & Palmer similarity is a metric to calculate the sematic similarity between 2 words/phrases based on the position of concepts $c_1$ and $c_2$ in the taxonomy, relative to the position of their **_Least Common Subsumer_** $LCS(c_1, c_2)$. *(In an directed acyclic graph, the Least Common Subsumer is the is the deepest node that has both the nodes under consideration as descendants, where we define each node to be a descendant of itself)*

WUP similarity works for single-word answers (& hence, we use if for our task), but doesn't work for phrases or sentences.

`nltk` has an implementation of Wu & Palmer similarity score based on the WordNet taxanomy. Here, we have adapted the [implementation of Wu & Palmer similarity as defined along with the DAQUAR dataset](https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/calculate_wups.py).
"""

def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a)
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score

def batch_wup_measure(labels, preds):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)

import nltk
nltk.download('wordnet')

labels = np.random.randint(len(answer_space), size=5)
preds = np.random.randint(len(answer_space), size=5)

def showAnswers(ids):
    print([answer_space[id] for id in ids])

showAnswers(labels)
showAnswers(preds)

print("Predictions vs Labels: ", batch_wup_measure(labels, preds))
print("Labels vs Labels: ", batch_wup_measure(labels, labels))

def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    preds = logits.argmax(axis=-1)
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }

"""## Model Training & Evaluation

### Define the Arguments needed for Training
"""

args = TrainingArguments(
    output_dir="checkpoint",
    seed=13573,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,             # Save only the last 3 checkpoints at any given time while training
    metric_for_best_model='wups',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    remove_unused_columns=False,
    num_train_epochs=20,
    fp16=True,
    # warmup_ratio=0.01,
    # learning_rate=5e-4,
    # weight_decay=1e-4,
    # gradient_accumulation_steps=2,
    dataloader_num_workers=8,
    load_best_model_at_end=True,
)

"""### Create the Multimodal Models using User-Defined Text/Image  Transformers & Train it on the Dataset"""

def createAndTrainModel(dataset, args, text_model=text_encoder, image_model=vision_encoder, multimodal_model='bert_vit'):
    collator, model = createMultimodalVQACollatorAndModel(text_model, image_model)

    multi_args = deepcopy(args)
    multi_args.output_dir = os.path.join("/kaggle/working/", "checkpoint-bert_vit", multimodal_model)
    multi_trainer = Trainer(
        model,
        multi_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    train_multi_metrics = multi_trainer.train()
    eval_multi_metrics = multi_trainer.evaluate()

    return collator, model, train_multi_metrics, eval_multi_metrics, multi_trainer
training_start_time = time.time()

import wandb
wandb.login(key="your_key_here")

collator, model, train_multi_metrics, eval_multi_metrics, multi_trainer = createAndTrainModel(dataset, args)
training_time = (time.time() - training_start_time)
print("training time= ", training_time)
#memory_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
#torch.cuda.reset_peak_memory_stats()
#print("Memory usage= ", memory_usage)
memory_usage_bytes = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
torch.cuda.reset_peak_memory_stats()
memory_usage_gb = memory_usage_bytes / (1024 ** 3)  # Convert bytes to GB
print("Memory usage (GB) = ", memory_usage_gb)

eval_multi_metrics

epochs = []
loss = []
eval_loss = []
eval_acc = []
eval_wups = []
eval_f1 = []


for i in range(0,len(multi_trainer.state.log_history),2):
    row = {**multi_trainer.state.log_history[i], **multi_trainer.state.log_history[i+1]}
    #print(row)
    epochs.append(row['epoch'])
    try:
        loss.append(row['loss'])
    except:
        loss.append(row['train_loss'])
    eval_loss.append(row['eval_loss'])
    eval_acc.append(row['eval_acc'])
    eval_f1.append(row['eval_f1'])
    eval_wups.append(row['eval_wups'])

# print(len(epochs), epochs)
# print(len(loss), loss)
# print(len(eval_loss), eval_loss)
# print(len(eval_acc), eval_acc)
# print(len(eval_wups), eval_wups)
# print(len(eval_f1), eval_f1)

from matplotlib.pylab import plt
# Plot and label the training and validation loss values
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, eval_loss, label='Validation Loss')

# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Set the tick locations
#plt.xticks(arange(0, 21, 2))

# Display the plot
plt.legend(loc='best')
file_path = "/kaggle/working/YesNo Dataset/bert_vit_add_batch16_epochs20_dim1536/plot.png"
plt.savefig(file_path)
plt.show()

import json
matrix = dict(eval_multi_metrics)
matrix['memory_usage_gb'] = memory_usage_gb
matrix['training_time'] = training_time
matrix['plot_source'] = dict()
matrix['plot_source']['epochs'] = epochs

matrix['plot_source']['loss'] =loss
matrix['plot_source']['eval_loss'] =eval_loss
matrix['plot_source']['eval_acc'] =eval_acc
matrix['plot_source']['eval_wups'] =eval_wups
matrix['plot_source']['eval_f1'] =eval_f1

def save_metrics_as_json(file_path):
    """
    Calls eval_multi_metrics to get metrics and saves them as a JSON file.

    Parameters:
        file_path (str): Path to save the JSON file.
    """
    metrics = matrix
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        print("Metrics saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
file_path = "/kaggle/working/YesNo Dataset/bert_vit_add_batch16_epochs20_dim1536/eval_results.json"
save_metrics_as_json(file_path)

"""## Examples of Model Inferencing

### Loading the Model from Checkpoint
"""

# Do not run this cell in colab instead use next cell
# model = MultimodalVQAModel()

# We use the checkpoint giving best results
# model.load_state_dict(torch.load(os.path.join("/content", "checkpoint", "bert_vit", "checkpoint-1500", "pytorch_model.bin")))
# model.to(device)

from safetensors.torch import load_file
import torch

# Define and load the model
model = MultimodalVQAModel()

# Load the model weights from model.safetensors
checkpoint_path = os.path.join("/kaggle/working", "checkpoint-bert_vit", "bert_vit", "checkpoint-400", "model.safetensors")
model_weights = load_file(checkpoint_path)
model.load_state_dict(model_weights)
model.to(device)

sample = collator(dataset["test"][100:105]) # change number of rows according to your dataset's test num_rows

input_ids = sample["input_ids"].to(device)
token_type_ids = sample["token_type_ids"].to(device)
attention_mask = sample["attention_mask"].to(device)
pixel_values = sample["pixel_values"].to(device)
labels = sample["labels"].to(device)

"""### Pass the Samples through the Model & inspect the Predictions"""

model.eval()
output = model(input_ids, pixel_values, attention_mask, token_type_ids, labels)

preds = output["logits"].argmax(axis=-1).cpu().numpy()
preds

for i in range(100, 105): # change number of rows according to your dataset's test num_rows
    print("*********************************************************")
    showExample(train=False, id=i)
    print("Predicted Answer:\t", answer_space[preds[i-105]]) # change i-[lower-range]
    print("*********************************************************")

"""## Inspecting Model Size"""

def countTrainableParameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("No. of trainable parameters:\t{0:,}".format(num_params))

countTrainableParameters(model) # For BERT-ViT model