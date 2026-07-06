#!/usr/bin/env python
import csv
import gzip
import io
import json
import os
import tarfile
import textwrap
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ocl.cli import train
from omegaconf import OmegaConf
from torchvision import transforms


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_DIR = os.path.join(
    REPO_ROOT,
    "outputs/projects/prompting/vg/"
    "prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip/"
    "2026-07-03_18-43-50_clip_layer2_tuned_5k_bs128_amp",
)
CONFIG_PATH = os.environ.get("CTRLO_CONFIG_PATH", os.path.join(RUN_DIR, "config/config.yaml"))
CHECKPOINT_PATH = os.environ.get(
    "CTRLO_CHECKPOINT_PATH", os.path.join(RUN_DIR, "checkpoints/epoch=1-step=5000.ckpt")
)
VAL_SHARD = os.environ.get(
    "CTRLO_VG_VAL_SHARD",
    os.path.join(REPO_ROOT, "scripts/datasets/outputs/vg_disjoint_clip/val/shard-000000.tar"),
)
OUTPUT_DIR = os.environ.get(
    "CTRLO_DIAG_OUTPUT_DIR",
    os.path.join(REPO_ROOT, "_001n_my/_009n_binding_slice_diagnostics"),
)
MAX_SAMPLES = int(os.environ.get("CTRLO_DIAG_MAX_SAMPLES", "0"))
BATCH_SIZE = int(os.environ.get("CTRLO_DIAG_BATCH_SIZE", "32"))
NUM_SLOTS = int(os.environ.get("CTRLO_NUM_SLOTS", "7"))
SMALL_AREA_THRESHOLD = float(os.environ.get("CTRLO_SMALL_AREA_THRESHOLD", "0.02"))
TOP_FAILURES = int(os.environ.get("CTRLO_TOP_FAILURES", "12"))
SELECTION_MODE = os.environ.get("CTRLO_DIAG_SELECTION", "first")

TEXT_TERMS = {
    "word",
    "text",
    "letter",
    "letters",
    "logo",
    "sign",
    "number",
    "writing",
    "label",
    "flag",
    "shirt",
    "jersey",
    "print",
    "printed",
    "advertising",
}
PART_TERMS = {
    "hand",
    "head",
    "face",
    "eye",
    "eyes",
    "leg",
    "legs",
    "arm",
    "arms",
    "wheel",
    "window",
    "door",
    "handle",
    "tail",
    "branch",
    "branches",
    "leaf",
    "leaves",
    "hair",
    "shoe",
    "helmet",
    "goggles",
    "rail",
}


def ensure_rgb_image(image):
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] > 3:
        return image[:, :, :3]
    return image


def load_samples(shard_path, max_samples):
    samples = []
    with tarfile.open(shard_path) as tar:
        keys = sorted({name.split(".")[0] for name in tar.getnames()})
        if max_samples > 0:
            keys = keys[:max_samples]
        for key in keys:
            image = np.load(io.BytesIO(gzip.decompress(tar.extractfile(f"{key}.image.npy.gz").read())))
            image = ensure_rgb_image(image)
            names = json.loads(tar.extractfile(f"{key}.name.json").read())
            mask = np.load(
                io.BytesIO(gzip.decompress(tar.extractfile(f"{key}.instance_mask.npy.gz").read()))
            )
            centroids = np.load(io.BytesIO(tar.extractfile(f"{key}.bbox_centroids.npy").read()))
            instance_bbox = np.load(io.BytesIO(tar.extractfile(f"{key}.instance_bbox.npy").read()))
            name_embedding = np.load(
                io.BytesIO(gzip.decompress(tar.extractfile(f"{key}.name_embedding.npy.gz").read()))
            ).astype("float32")
            samples.append(
                {
                    "key": key,
                    "image": image,
                    "names": names,
                    "mask": mask,
                    "centroids": centroids,
                    "instance_bbox": instance_bbox,
                    "name_embedding": name_embedding,
                }
            )
    return samples


def pad_list(values, fill, length):
    values = list(values[:length])
    return values + [fill] * (length - len(values))


def phrase_has_term(phrase, terms):
    normalized = phrase.lower().replace("-", " ")
    tokens = {token.strip(".,;:!?()[]{}'\"") for token in normalized.split()}
    return bool(tokens & terms)


def slice_names(phrase, area_frac):
    slices = ["overall"]
    if area_frac > 0 and area_frac < SMALL_AREA_THRESHOLD:
        slices.append("small")
    if phrase_has_term(phrase, TEXT_TERMS):
        slices.append("text")
    if phrase_has_term(phrase, PART_TERMS):
        slices.append("part")
    return slices


def bbox_area_fraction(bbox):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return (width * height) / (224.0 * 224.0)


def hard_score(sample, source_idx):
    phrase = sample["names"][source_idx]
    area_frac = bbox_area_fraction(sample["instance_bbox"][source_idx])
    is_small = 0.0 < area_frac < SMALL_AREA_THRESHOLD
    is_text = phrase_has_term(phrase, TEXT_TERMS)
    is_part = phrase_has_term(phrase, PART_TERMS)
    score = 1.0
    if is_small:
        score += 6.0
    if is_text:
        score += 5.0
    if is_part:
        score += 4.0
    return score


def select_source_indices(sample):
    count = min(len(sample["names"]), len(sample["name_embedding"]))
    indices = list(range(count))
    if SELECTION_MODE == "first":
        return indices[:NUM_SLOTS]
    if SELECTION_MODE == "hard":
        return sorted(indices, key=lambda idx: (-hard_score(sample, idx), idx))[:NUM_SLOTS]
    raise ValueError(f"Unsupported CTRLO_DIAG_SELECTION={SELECTION_MODE!r}")


def build_batch(samples, image_transform, device):
    images = torch.stack([image_transform(sample["image"]) for sample in samples]).to(device)
    selected_source_indices = [select_source_indices(sample) for sample in samples]
    prompts = [
        pad_list([sample["names"][idx] for idx in source_indices], "other", NUM_SLOTS)
        for sample, source_indices in zip(samples, selected_source_indices)
    ]
    name_embeddings = torch.tensor(
        [
            pad_list(
                [sample["name_embedding"][idx].tolist() for idx in source_indices],
                [0.0] * 512,
                NUM_SLOTS,
            )
            for sample, source_indices in zip(samples, selected_source_indices)
        ],
        dtype=torch.float32,
        device=device,
    )
    contrastive_loss_mask = torch.tensor(
        [[1 if prompt != "other" else 0 for prompt in row] for row in prompts],
        dtype=torch.long,
        device=device,
    )
    bbox_centroids = torch.tensor(
        [
            pad_list([sample["centroids"][idx].tolist() for idx in source_indices], [-1, -1], NUM_SLOTS)
            for sample, source_indices in zip(samples, selected_source_indices)
        ],
        dtype=torch.float32,
        device=device,
    )
    instance_bbox = torch.tensor(
        [
            pad_list(
                [sample["instance_bbox"][idx].tolist() for idx in source_indices],
                [-1, -1, -1, -1],
                NUM_SLOTS,
            )
            for sample, source_indices in zip(samples, selected_source_indices)
        ],
        dtype=torch.float32,
        device=device,
    )
    inputs = {
        "image": images,
        "bbox_centroids": bbox_centroids / 224.0,
        "contrastive_loss_mask": contrastive_loss_mask,
        "name_embedding": name_embeddings,
        "instance_bbox": instance_bbox / 224.0,
        "batch_size": len(samples),
    }
    return prompts, selected_source_indices, inputs


def compute_record(sample, slot_idx, source_idx, phrase, pred_mask):
    label_id = source_idx + 1
    gt_mask = sample["mask"] == label_id
    gt_area = int(gt_mask.sum())
    image_area = int(gt_mask.size)
    area_frac = gt_area / image_area if image_area else 0.0

    pred = pred_mask.astype(np.float32)
    pred_sum = float(pred.sum())
    intersection = float((pred * gt_mask.astype(np.float32)).sum())
    union = float(pred_sum + gt_area - intersection)
    soft_iou = intersection / union if union > 0 else 0.0
    mass_in_gt = intersection / pred_sum if pred_sum > 0 else 0.0
    gt_coverage = intersection / gt_area if gt_area > 0 else 0.0
    peak_y, peak_x = np.unravel_index(int(pred.argmax()), pred.shape)
    peak_in_gt = bool(gt_mask[peak_y, peak_x]) if gt_area > 0 else False

    return {
        "sample_id": sample["key"],
        "slot_idx": slot_idx,
        "source_idx": source_idx,
        "phrase": phrase,
        "gt_area": gt_area,
        "area_frac": area_frac,
        "soft_iou": soft_iou,
        "mass_in_gt": mass_in_gt,
        "gt_coverage": gt_coverage,
        "peak_in_gt": int(peak_in_gt),
        "pred_sum": pred_sum,
        "slices": slice_names(phrase, area_frac),
    }


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        for name in record["slices"]:
            groups[name].append(record)

    summary = {}
    for name, items in sorted(groups.items()):
        summary[name] = {
            "n": len(items),
            "soft_iou_mean": float(np.mean([item["soft_iou"] for item in items])) if items else 0.0,
            "mass_in_gt_mean": float(np.mean([item["mass_in_gt"] for item in items])) if items else 0.0,
            "gt_coverage_mean": float(np.mean([item["gt_coverage"] for item in items])) if items else 0.0,
            "peak_in_gt_rate": float(np.mean([item["peak_in_gt"] for item in items])) if items else 0.0,
            "area_frac_mean": float(np.mean([item["area_frac"] for item in items])) if items else 0.0,
        }
    return summary


def colorize_mask(mask):
    palette = np.array(
        [
            [0, 0, 0],
            [230, 57, 70],
            [42, 157, 143],
            [69, 123, 157],
            [233, 196, 106],
            [155, 93, 229],
            [244, 162, 97],
            [46, 204, 113],
            [231, 111, 81],
            [91, 192, 190],
            [247, 37, 133],
        ],
        dtype=np.float32,
    )
    return palette[np.asarray(mask, dtype=np.int64) % len(palette)] / 255.0


def overlay_binary(image, mask, color):
    image_float = image.astype(np.float32) / 255.0
    color_arr = np.zeros_like(image_float)
    for channel, value in enumerate(color):
        color_arr[:, :, channel] = mask.astype(np.float32) * value
    return np.clip(0.55 * image_float + 0.45 * color_arr, 0.0, 1.0)


def overlay_soft(image, pred):
    image_float = image.astype(np.float32) / 255.0
    heat = cv2.applyColorMap(np.uint8(np.clip(pred, 0, 1) * 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.55 * image_float + 0.45 * heat, 0.0, 1.0)


def save_failures(records, sample_by_id, pred_by_key_slot, output_path):
    slice_order = ["small", "text", "part"]
    rows = []
    for slice_name in slice_order:
        candidates = [record for record in records if slice_name in record["slices"]]
        candidates = sorted(candidates, key=lambda item: (item["soft_iou"], item["mass_in_gt"]))
        rows.extend([(slice_name, record) for record in candidates[:TOP_FAILURES]])

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for row_idx, (slice_name, record) in enumerate(rows):
        sample = sample_by_id[record["sample_id"]]
        image = sample["image"]
        gt_mask = sample["mask"] == (record["source_idx"] + 1)
        pred = pred_by_key_slot[(record["sample_id"], record["slot_idx"])]
        image_224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"{slice_name}\\n{record['sample_id']}", fontsize=8)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(0.55 * image.astype(np.float32) / 255.0 + 0.45 * colorize_mask(sample["mask"]))
        axes[row_idx, 1].set_title("GT all", fontsize=8)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay_binary(image_224, gt_mask, (1.0, 0.0, 0.0)))
        axes[row_idx, 2].set_title("GT phrase", fontsize=8)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(overlay_soft(image_224, pred))
        title = (
            f"IoU {record['soft_iou']:.3f} mass {record['mass_in_gt']:.3f}\\n"
            + textwrap.fill(record["phrase"], 24)
        )
        axes[row_idx, 3].set_title(title, fontsize=8)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_successes(records, sample_by_id, pred_by_key_slot, output_path):
    slice_order = ["small", "text", "part"]
    rows = []
    for slice_name in slice_order:
        candidates = [record for record in records if slice_name in record["slices"]]
        candidates = sorted(
            candidates,
            key=lambda item: (item["peak_in_gt"], item["soft_iou"], item["mass_in_gt"]),
            reverse=True,
        )
        rows.extend([(slice_name, record) for record in candidates[:TOP_FAILURES]])

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for row_idx, (slice_name, record) in enumerate(rows):
        sample = sample_by_id[record["sample_id"]]
        image = sample["image"]
        gt_mask = sample["mask"] == (record["source_idx"] + 1)
        pred = pred_by_key_slot[(record["sample_id"], record["slot_idx"])]
        image_224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"{slice_name}\\n{record['sample_id']}", fontsize=8)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(0.55 * image.astype(np.float32) / 255.0 + 0.45 * colorize_mask(sample["mask"]))
        axes[row_idx, 1].set_title("GT all", fontsize=8)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay_binary(image_224, gt_mask, (1.0, 0.0, 0.0)))
        axes[row_idx, 2].set_title("GT phrase", fontsize=8)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(overlay_soft(image_224, pred))
        title = (
            f"IoU {record['soft_iou']:.3f} mass {record['mass_in_gt']:.3f}\\n"
            + textwrap.fill(record["phrase"], 24)
        )
        axes[row_idx, 3].set_title(title, fontsize=8)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load(CONFIG_PATH)
    model = train.build_model_from_config(config, CHECKPOINT_PATH).to(device)
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    samples = load_samples(VAL_SHARD, MAX_SAMPLES)
    records = []
    pred_by_key_slot = {}
    sample_by_id = {sample["key"]: sample for sample in samples}

    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start : start + BATCH_SIZE]
        prompts, selected_source_indices, inputs = build_batch(batch, image_transform, device)
        outputs = model(inputs)
        pred_masks = outputs["object_decoder"].masks_as_image.detach().cpu().view(len(batch), NUM_SLOTS, 224, 224).numpy()

        for batch_idx, sample in enumerate(batch):
            for slot_idx, phrase in enumerate(prompts[batch_idx]):
                if phrase == "other" or slot_idx >= len(selected_source_indices[batch_idx]):
                    continue
                source_idx = selected_source_indices[batch_idx][slot_idx]
                pred = pred_masks[batch_idx, slot_idx]
                pred_by_key_slot[(sample["key"], slot_idx)] = pred
                records.append(compute_record(sample, slot_idx, source_idx, phrase, pred))

    summary = aggregate(records)

    csv_path = os.path.join(OUTPUT_DIR, "binding_slice_records.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "sample_id",
            "slot_idx",
            "source_idx",
            "phrase",
            "gt_area",
            "area_frac",
            "soft_iou",
            "mass_in_gt",
            "gt_coverage",
            "peak_in_gt",
            "pred_sum",
            "slices",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["slices"] = "|".join(record["slices"])
            writer.writerow(row)

    json_path = os.path.join(OUTPUT_DIR, "binding_slice_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(OUTPUT_DIR, "binding_slice_summary.md")
    with open(md_path, "w") as f:
        f.write("# Binding Slice Diagnostics\n\n")
        f.write(f"checkpoint: `{CHECKPOINT_PATH}`\n\n")
        f.write(f"val_shard: `{VAL_SHARD}`\n\n")
        f.write(f"selection: `{SELECTION_MODE}`\n\n")
        f.write(f"samples: `{len(samples)}`\n\n")
        f.write("| slice | n | area mean | soft IoU | mass in GT | GT coverage | peak in GT |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for name in ["overall", "small", "text", "part"]:
            item = summary.get(name, {})
            f.write(
                f"| {name} | {item.get('n', 0)} | {item.get('area_frac_mean', 0):.4f} | "
                f"{item.get('soft_iou_mean', 0):.4f} | {item.get('mass_in_gt_mean', 0):.4f} | "
                f"{item.get('gt_coverage_mean', 0):.4f} | {item.get('peak_in_gt_rate', 0):.4f} |\n"
            )

    failure_path = os.path.join(OUTPUT_DIR, "binding_slice_failures.png")
    save_failures(records, sample_by_id, pred_by_key_slot, failure_path)
    success_path = os.path.join(OUTPUT_DIR, "binding_slice_successes.png")
    save_successes(records, sample_by_id, pred_by_key_slot, success_path)

    print(f"records: {csv_path}")
    print(f"summary_json: {json_path}")
    print(f"summary_md: {md_path}")
    print(f"failures: {failure_path}")
    print(f"successes: {success_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
