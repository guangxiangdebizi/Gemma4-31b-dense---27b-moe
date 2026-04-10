from modelscope import snapshot_download
import time

start = time.time()
print(f"[31B-it] Start downloading...", flush=True)
snapshot_download(
    'google/gemma-4-31b-it',
    local_dir='/root/autodl-tmp/MyProject/model/gemma-4-31B-it'
)
elapsed = (time.time() - start) / 60
print(f"[31B-it] DONE! took {elapsed:.1f} min", flush=True)
