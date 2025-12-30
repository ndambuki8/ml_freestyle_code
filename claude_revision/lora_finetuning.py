from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# Load base model with quantization
momdel = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# lora connfig

lora_config = LoraConfig(
    r=16, #rankn
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir="./lora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    optim="paged_adamw_8bit"
)

# Train with custom dataset
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset,
    dataset_text_F=field="text",
    max_seq_length=512
)

trainer.train()
model.save_pretrained("./lora-finetuned")