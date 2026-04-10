import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from huggingface_hub import snapshot_download as hf_download
import time

start = time.time()
print("[26B-A4B-it-uncensored] Start downloading...", flush=True)

hf_download(
    repo_id='TrevorJS/gemma-4-26B-A4B-it-uncensored',
    local_dir='/root/autodl-tmp/MyProject/model/gemma-4-26B-A4B-it-uncensored',
    max_workers=4,
    resume_download=True,
)

elapsed = (time.time() - start) / 60
print(f"[26B-A4B-it-uncensored] DONE! took {elapsed:.1f} min", flush=True)
