import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.integrations.moe as moe_module

# RTX PRO 6000 不支持 grouped_mm (需要 SM90/Hopper)，强制走 fallback 路径
moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

MODEL_PATH = "/root/autodl-tmp/MyProject/model/gemma-4-26B-A4B-it"
GENERATION_CONFIG = dict(
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True,
)

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading MoE model (BF16)... this may take a few minutes", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "90GiB"},
)
model.eval()
print("Model loaded!\n", flush=True)

test_prompts = [
    "用简洁的中文介绍一下你自己，你是谁，你能做什么？",
    "请用Python写一个快速排序算法，并解释时间复杂度。",
    "如果我有一个装满水的杯子，把它倒扣在桌子上，然后在杯底打一个洞，会发生什么？请一步步分析。",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"{'='*60}", flush=True)
    print(f"Test {i}: {prompt}", flush=True)
    print(f"{'='*60}", flush=True)

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    print(f"[Input tokens: {input_len}]", flush=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\n[Gemma 4 MoE Response] ({len(new_tokens)} tokens):\n", flush=True)
    print(response, flush=True)
    print("\n", flush=True)

print("All tests done!", flush=True)
