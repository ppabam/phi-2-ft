import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with PEFT LoRA")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/phi-2")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./models/lora-phi2")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # 2) LoRA 설정
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    # 3) 데이터셋 로드 (jsonl, line by line)
    dataset = load_dataset('json', data_files={"train": args.train_file})

    # 4) 토큰화 함수
    def tokenize_fn(example):
        # 질문-답변 포맷으로 묶기 (예시)
        prompt = f"### 질문: {example['instruction']}\n### 답변: {example['output']}"
        tokens = tokenizer(prompt, max_length=args.max_seq_length, truncation=True, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_fn, batched=False)

    # 5) 트레이닝 아규먼트 세팅
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        fp16=True,
        evaluation_strategy="no",
        remove_unused_columns=False,
    )

    # 6) Trainer 생성 및 학습 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
