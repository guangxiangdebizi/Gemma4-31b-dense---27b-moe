from modelscope import snapshot_download
import time

start = time.time()
print(f"[26B-A4B-it] Start downloading...", flush=True)
snapshot_download(
    'google/gemma-4-26B-A4B-it',
    local_dir='/root/autodl-tmp/MyProject/model/gemma-4-26B-A4B-it'
)
elapsed = (time.time() - start) / 60
print(f"[26B-A4B-it] DONE! took {elapsed:.1f} min", flush=True)
