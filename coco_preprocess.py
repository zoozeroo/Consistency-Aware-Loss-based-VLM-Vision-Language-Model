import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 병렬 처리용 함수
def process_image_safe(image_id, captions, bboxes, category_id_to_name):
    import spacy
    import inflect

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    p = inflect.engine()

    found_tokens = set()
    for doc in nlp.pipe(captions, batch_size=32):
        for token in doc:
            if token.pos_ == "NOUN":
                word = token.text.lower()
                singular = p.singular_noun(word) or word
                found_tokens.add(singular)

    matched_entries = []
    for bbox_entry in bboxes:
        category_id = bbox_entry["category_id"]
        label = category_id_to_name[category_id].lower()
        label_singular = p.singular_noun(label) or label
        if label_singular in found_tokens:
            matched_entries.append({
                "token": label_singular,
                "bbox": bbox_entry["bbox"],
                "label": label
            })

    return {
        "image_id": image_id,
        "captions": captions,
        "matches": matched_entries
    }

# 🎯 핵심: lambda 대신 사용할 helper 함수
def process_wrapper(args):
    return process_image_safe(*args)

def main():
    captions_path = "annotations_trainval2017/annotations/captions_train2017.json"
    instances_path = "annotations_trainval2017/annotations/instances_train2017.json"
    output_path = "coco_token_bbox_matched.json"

    with open(captions_path, "r") as f:
        captions_data = json.load(f)
    with open(instances_path, "r") as f:
        instances_data = json.load(f)

    category_id_to_name = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
    image_id_to_captions = defaultdict(list)
    image_id_to_bboxes = defaultdict(list)

    for ann in captions_data["annotations"]:
        image_id_to_captions[ann["image_id"]].append(ann["caption"])
    for ann in instances_data["annotations"]:
        image_id_to_bboxes[ann["image_id"]].append({
            "category_id": ann["category_id"],
            "bbox": ann["bbox"]
        })

    inputs = [
        (
            image_id,
            image_id_to_captions[image_id],
            image_id_to_bboxes.get(image_id, []),
            category_id_to_name,
        )
        for image_id in image_id_to_captions
    ]

    print(f"총 작업 수: {len(inputs)}개. 병렬 처리 시작...")

    results = []
    with Pool(processes=min(cpu_count(), 8)) as pool:
        for r in tqdm(pool.imap_unordered(process_wrapper, inputs), total=len(inputs)):
            results.append(r)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    matched_count = sum(1 for r in results if r["matches"])
    unmatched_count = len(results) - matched_count
    total = matched_count + unmatched_count

    print("총 이미지 수:", total)
    print(f"매칭된 이미지 수: {matched_count} ({matched_count / total:.2%})")
    print(f"매칭 안된 이미지 수: {unmatched_count} ({unmatched_count / total:.2%})")
    print(f"결과 저장 위치: {output_path}")

if __name__ == "__main__":
    main()
