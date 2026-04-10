"""
统计 Gemma 4 26B-A4B MoE 在不同输入下 Router 选中的专家 ID（0~127）。

说明：
- 官方未给每个专家命名；以下为「路由统计」，不是语义标签。
- Blackwell 等 GPU 上默认 grouped_mm 可能报错，需设 text_config._experts_implementation = "eager"。
"""
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/autodl-tmp/MyProject/model/gemma-4-26B-A4B-it"

TEST_INPUTS = {
    "中文对话": "你好，请用一句话介绍量子纠缠。",
    "英文对话": "Explain gradient descent in one sentence.",
    "数学": "Solve: integral of exp(-x^2) from -inf to inf.",
    "Python": "def fib(n):\n    # complete recursive fibonacci",
    "日语": "東京の天気について一文で述べてください。",
    "JSON": '{"user":{"id":1,"tags":["a","b"]}} 请解释这个结构',
}

hooks = []
captured = []


def router_hook(_module, _inp, out):
    # Gemma4TextRouter.forward -> (router_probabilities, top_k_weights, top_k_index)
    top_k_index = out[2]
    captured.append(top_k_index.detach().cpu().flatten().tolist())


def main():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model (CUDA, BF16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.config.text_config._experts_implementation = "eager"

    for i in range(30):
        h = model.model.language_model.layers[i].router.register_forward_hook(router_hook)
        hooks.append(h)

    print("Hooks:", len(hooks), "\n", flush=True)

    for label, text in TEST_INPUTS.items():
        captured.clear()
        messages = [{"role": "user", "content": text}]
        batch = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            model(**batch)

        flat = [x for chunk in captured for x in chunk]
        cnt = Counter(flat)
        top15 = cnt.most_common(15)
        seq_len = len(captured[0]) // 8 if captured else 0

        print("=" * 60, flush=True)
        print(f"场景: {label}", flush=True)
        print(f"文本: {text[:60]}...", flush=True)
        print(f"MoE 层数: {len(captured)} | 序列长度(位置数): {seq_len}", flush=True)
        print(f"Top-15 专家 ID（出现次数）: {top15}", flush=True)
        print(flush=True)

    for h in hooks:
        h.remove()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
