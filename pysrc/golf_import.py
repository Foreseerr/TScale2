import os
import struct
from pathlib import Path
from huggingface_hub import hf_hub_download
import json

DST_FOLDER = 'D:/text/golf'


def process(example, file):
    # UTF-8 encoding
    text = example['text'].encode()
    
    # write len as 32-bit unsigned integer
    data_len = len(text)
    file.write(struct.pack('I', data_len))
    # write text
    file.write(text)


if __name__ == '__main__':
    jsonl_path = hf_hub_download(
        repo_id="willdepueoai/parameter-golf", 
        filename="docs_selected.jsonl",
        subfolder="datasets",
        repo_type="dataset"
    )

    part_id = 0
    doc_count = 0
    file = None

    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if file is None or doc_count > 50000:
                print(f'write part {part_id}')
                if file:
                    file.close()
                part_root = Path(DST_FOLDER + f'/{part_id}/').expanduser().resolve()
                part_root.mkdir(parents=True, exist_ok=True)
                file = open(f'{DST_FOLDER}/{part_id}/0.bin', 'wb')
                part_id += 1
                doc_count = 0

            process(json.loads(line), file)
            doc_count += 1

    if file:
        file.close()
