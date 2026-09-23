"""Minimal deterministic Transformers inference example."""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "OmTheLast/taj-mahal-history-tutor-4b-experimental"
SYSTEM = (
    "You are a history tutor. Answer each question directly using the requested "
    "format. Give accurate information and do not invent facts."
)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Where is the Taj Mahal?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(inputs, max_new_tokens=128, do_sample=False)
    print(tokenizer.decode(output[0, inputs.shape[-1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()

