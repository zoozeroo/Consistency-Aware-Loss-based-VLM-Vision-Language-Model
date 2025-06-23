#weekly report week9의 코드(clip의 downstream task 성능체크)
import torch
import torch.nn.functional as F
from transformers import (
    CLIPProcessor, CLIPModel
)
from PIL import Image
import json
import os
from tqdm import tqdm
from evaluate import load as load_metric
from collections import defaultdict
import random
from pycocoevalcap.cider.cider import Cider

# ==========================
# 모델 로딩
# ==========================

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ==========================
# 데이터 경로 설정
# ==========================

COCO_IMAGE_DIR = "./val2017"
COCO_ANN_PATH = "annotations_trainval2017/annotations/captions_val2017.json"

# 데이터 다운로드
# https://cocodataset.org/#download
# 2017 Val images [5K/1GB]
# 2017 Train/Val annotations [241MB]

# ==========================
# 캡션 로딩
# ==========================

with open(COCO_ANN_PATH, 'r') as f:
    coco_data = json.load(f)

id_to_captions = defaultdict(list)
for ann in coco_data['annotations']:
    id_to_captions[ann['image_id']].append(ann['caption'])

id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

# 전체 캡션 중에서 open-domain caption pool 구성
caption_pool = []
for caps in id_to_captions.values():
    caption_pool.extend(caps)
caption_pool = list(set(caption_pool))  # 중복 제거
random.shuffle(caption_pool)
caption_pool_subset = caption_pool[:5000]  # open-domain caption pool for captioning

# ==========================
# 이미지-텍스트 Retrieval 평가 (Closed-domain)
# ==========================

valid_ids = []
for img_id in id_to_filename:
    file_path = os.path.join(COCO_IMAGE_DIR, id_to_filename[img_id])
    if os.path.exists(file_path):
        valid_ids.append(img_id)

selected_ids = valid_ids[:100]
image_embeds_list = []
text_embeds_list = []
text_list = []

for img_id in tqdm(selected_ids, desc="Encoding image-text pairs"):
    file_path = os.path.join(COCO_IMAGE_DIR, id_to_filename[img_id])
    image = Image.open(file_path).convert("RGB")
    caption = id_to_captions[img_id][0]  # closed-domain caption (GT)
    inputs = clip_processor(images=image, text=[caption], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    image_embeds_list.append(outputs.image_embeds)
    text_embeds_list.append(outputs.text_embeds)
    text_list.append(caption)

image_embeds_all = torch.cat(image_embeds_list, dim=0)
text_embeds_all = torch.cat(text_embeds_list, dim=0)

image_embeds_all = F.normalize(image_embeds_all, dim=-1)
text_embeds_all = F.normalize(text_embeds_all, dim=-1)

sim_matrix = torch.matmul(image_embeds_all, text_embeds_all.T)

def compute_recall(sim_matrix, k):
    correct = 0
    for i in range(sim_matrix.size(0)):
        topk = sim_matrix[i].topk(k).indices
        if i in topk:
            correct += 1
    return correct / sim_matrix.size(0)

recall1 = compute_recall(sim_matrix, 1)
recall5 = compute_recall(sim_matrix, 5)
recall10 = compute_recall(sim_matrix, 10)

print("\n[Image-Text Retrieval Results (Closed-domain)]")
print(f"Recall@1: {recall1:.4f}")
print(f"Recall@5: {recall5:.4f}")
print(f"Recall@10: {recall10:.4f}")

# ==========================
# 간접 방식 Captioning 평가 (Open-domain, CIDEr)
# ==========================

print("\n[Indirect Captioning Evaluation - CIDEr Metric (Open-domain)]")
# 이미지 임베딩 재사용: image_embeds_all

# open-domain caption pool embedding
text_embeds = []
for i in tqdm(range(0, len(caption_pool_subset), 64), desc="Encoding open-domain captions"):
    batch = caption_pool_subset[i:i+64]
    inputs = clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        txt_embed = clip_model.get_text_features(**inputs)
    text_embeds.append(txt_embed)

text_embeds_all = torch.cat(text_embeds, dim=0)
text_embeds_all = F.normalize(text_embeds_all, dim=-1)

sim_matrix_captioning = torch.matmul(image_embeds_all, text_embeds_all.T)

cider = Cider()
pred_dict = {}
ref_dict = {}

for i, img_id in enumerate(selected_ids[:20]):
    best_idx = sim_matrix_captioning[i].argmax().item()
    pred_caption = caption_pool_subset[best_idx]
    ref_captions = id_to_captions[img_id][:5]
    pred_dict[str(i)] = [pred_caption]
    ref_dict[str(i)] = ref_captions

score, _ = cider.compute_score(ref_dict, pred_dict)
print(f"CIDEr: {score:.4f}")

# ==========================
# 간접 방식 VQA 평가 예시
# ==========================

print("\n[Indirect VQA Evaluation - Matching Accuracy]")
custom_vqa = [
    {"image": "000000179765.jpg", "question": "What is the vehicle?", "choices": ["motorcycle", "bicycle", "car"]},
    {"image": "000000331352.jpg", "question": "Is this a bathroom?", "choices": ["yes", "no"]},
    {"image": "000000517069.jpg", "question": "Where is the person sitting?", "choices": ["on a bench", "on a sofa", "on the ground"]},
]

correct = 0
for sample in custom_vqa:
    image_path = os.path.join(COCO_IMAGE_DIR, sample["image"])
    image = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(images=image, return_tensors="pt")
    ques_inputs = clip_processor(text=[sample["question"]], return_tensors="pt")

    with torch.no_grad():
        img_embed = clip_model.get_image_features(**img_inputs)
        question_embed = clip_model.get_text_features(**ques_inputs)

    choice_embeds = []
    for ans in sample["choices"]:
        with torch.no_grad():
            ans_embed = clip_model.get_text_features(**clip_processor(text=[ans], return_tensors="pt"))
        choice_embeds.append(ans_embed)

    choice_embeds_all = torch.cat(choice_embeds, dim=0)
    question_embed = F.normalize(question_embed, dim=-1)
    choice_embeds_all = F.normalize(choice_embeds_all, dim=-1)

    sim = torch.matmul(question_embed, choice_embeds_all.T)
    pred_idx = torch.argmax(sim, dim=-1).item()
    pred_answer = sample["choices"][pred_idx]

    print(f"Image: {sample['image']}\nQ: {sample['question']}\nA: {pred_answer}\n")

    if "answer" in sample and pred_answer == sample["answer"]:
        correct += 1

if any("answer" in s for s in custom_vqa):
    acc = correct / sum("answer" in s for s in custom_vqa)
    print(f"Accuracy: {acc:.2%}")
else:
    print("(Ground truth 정답이 포함되어 있지 않아 Accuracy 계산 생략)")
