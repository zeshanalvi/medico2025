from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import os, json

# Output folder
d_path = "D:\\datasets\\medico_test\\"
img_dir = Path(os.path.abspath(os.path.join(d_path, "images")))
img_dir.mkdir(exist_ok=True, parents=True)

# Download original images once from SimulaMet-HOST/Kvasir-VQA
ds_host = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw")
_, idx = np.unique(ds_host["img_id"], return_index=True)
ds = ds.select(sorted(idx))
existing = set(p.stem for p in img_dir.glob("*.jpg"))
for row in tqdm(ds, desc="Saving unique images"):
    if row["img_id"] in existing: 
        continue
    row["image"].save(img_dir / f"{row['img_id']}.jpg")

# Save VLM-ready JSONLs (pointing to ORIGINAL images)
for split in ["train", "test"]:
    with open(f"{d_path}/Kvasir-VQA-x1-{split}.jsonl", "w", encoding="utf-8") as f:
        for r in load_dataset("SimulaMet/Kvasir-VQA-x1", split=split):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": f"<image>{r['question']}"},
                    {"role": "assistant", "content": r["answer"]}
                ],
                "images": [str(img_dir / f"{r['img_id']}.jpg")]
            }, ensure_ascii=False) + "\n")