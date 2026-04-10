import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/MyProject/model/gemma-4-31B-it"
MAX_CONTEXT = 32768
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

print("Loading model (BF16)... this may take a few minutes", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "90GiB"},
)
model.eval()
print("Model loaded! Ready to chat.\n", flush=True)
print("=" * 60)
print("Gemma 4 31B-it | BF16 | 32K context | thinking OFF")
print("Type 'quit' to exit, 'clear' to reset history")
print("=" * 60)

messages = []

while True:
    try:
        user_input = input("\n[You]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("Bye!")
        break
    if user_input.lower() == "clear":
        messages = []
        print("[System] History cleared.")
        continue

    messages.append({"role": "user", "content": user_input})

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    if input_ids.shape[1] > MAX_CONTEXT:
        print(f"[Warning] Input too long ({input_ids.shape[1]} tokens), truncating history...")
        messages = messages[-2:]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

    print("\n[Gemma 4]: ", end="", flush=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(response, flush=True)

    messages.append({"role": "assistant", "content": response})
