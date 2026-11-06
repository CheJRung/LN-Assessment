import argparse
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MODEL = PROJECT / "best"
SCORE_THRESHOLD = 0.75

os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT / ".cache"))

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy.sparse.csgraph import connected_components
import torch
from torch.nn import functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

def predictions(output, image_id, size):
    logits = output.masks_queries_logits[0].float()
    class_scores = output.class_queries_logits[0].float().softmax(dim=-1)[:, 0]
    foreground = logits > 0
    pixels = foreground.sum(dim=(1, 2))
    confidence = class_scores * (logits.sigmoid() * foreground).sum(dim=(1, 2)) / pixels.clamp_min(1)

    results = []
    for indices in pixels.nonzero(as_tuple=True)[0].split(8):
        if indices.numel() == 0:
            continue
        resized = F.interpolate(logits[None, indices], size=size, mode="bilinear",
                                align_corners=False)[0]
        masks = (resized > 0).permute(1, 2, 0).cpu().numpy()
        rles = coco_mask.encode(np.asfortranarray(masks, dtype=np.uint8))
        for rle, score in zip(rles, confidence[indices].tolist()):
            if coco_mask.area(rle):
                results.append({
                    "image_id": image_id, "category_id": 1, "segmentation": rle, "score": score,
                })
    return results


def merge_predictions(detections):
    masks = [
        detection["segmentation"]
        for detection in detections
        if detection["score"] >= SCORE_THRESHOLD
    ]
    if not masks:
        return 0, []

    overlap = coco_mask.iou(masks, masks, [0] * len(masks)) > 0
    group_count, labels = connected_components(overlap, directed=False)
    groups = [[] for _ in range(group_count)]
    for mask, label in zip(masks, labels):
        groups[label].append(mask)
    return len(masks), [coco_mask.merge(group) for group in groups]


def draw_contours(image, masks):
    overlay = np.array(image)
    for mask in masks:
        contours, _ = cv2.findContours(
            coco_mask.decode(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=3)
    return Image.fromarray(overlay)


def main():
    parser = argparse.ArgumentParser(description="Draw red Mask2Former instance contours.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL,
                        help="Local fine-tuned Mask2Former directory (default: best/).")
    parser.add_argument("--output", type=Path, help="PNG path inside the project directory.")
    args = parser.parse_args()

    source = args.image.resolve()
    default_output = PROJECT / "demo_outputs" / f"{source.stem}_contour.png"
    output = (args.output or default_output).resolve()
    if output.suffix.lower() != ".png" or not output.is_relative_to(PROJECT) or output == source:
        parser.error("--output must be a PNG path inside the project, separate from the input")

    with Image.open(source) as original:
        image = original.convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Mask2FormerImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model, local_files_only=True
    ).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            result = model(**inputs)
        detections = predictions(result, image_id=1, size=(image.height, image.width))

    instance_count, masks = merge_predictions(detections)
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_contours(image, masks).save(output)
    print(f"Instances (score >= {SCORE_THRESHOLD:.0%}): {instance_count}")
    print(f"Groups after merging: {len(masks)}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
