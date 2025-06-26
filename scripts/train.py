import os
import argparse
import torch # torch import 추가
from datasets import load_dataset
# ★★★ BitsAndBytesConfig 추가 ★★★
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training # ★★★ prepare_model_for_kbit_training 추가 ★★★

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with PEFT LoRA")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/phi-2")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./models/qlora-phi2")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1) # ★★★ 배치 사이즈 1로 변경 ★★★
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8) # ★★★ 그래디언트 누적 추가 ★★★
    parser.add_argument("--learning_rate", type=float, default=2e-4) # QLoRA는 learning rate를 조금 더 높게 설정하기도 함
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()

    # ★★★ 1. QLoRA를 위한 BitsAndBytesConfig 설정 ★★★
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2) 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Phi-2는 right-padding을 권장

    # ★★★ 모델 로드 시 양자화 설정(quantization_config) 적용 ★★★
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map={"": 0}, # GPU 0에 모델을 올림
        trust_remote_code=True,
    )
    
    # ★★★ 그래디언트 체크포인팅을 위해 모델 준비 ★★★
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 3) LoRA 설정
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16, # r과 lora_alpha는 조절 가능
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=[ # Phi-2의 target_modules, 모델마다 다를 수 있음
            "Wqkv",
            "fc1",
            "fc2",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4) 데이터셋 로드 및 토큰화
    dataset = load_dataset('json', data_files={"train": args.train_file})

    def tokenize_fn(examples):
        prompts = [f"### 질문: {instr}\n### 답변: {out}{tokenizer.eos_token}" for instr, out in zip(examples["instruction"], examples["output"])]
        tokens = tokenizer(
            prompts,
            max_length=args.max_seq_length,
            truncation=True,
            padding=False, # DataCollator에서 패딩하므로 False로 설정
            return_tensors=None,
        )
        tokens["labels"] = [input_ids.copy() for input_ids in tokens["input_ids"]]
        return tokens

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=list(dataset["train"].features)
    )

    # 5) 트레이닝 아규먼트 세팅
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, # ★★★ 누적 스텝 적용 ★★★
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True, # bfloat16을 사용할 수 있으면 bf16=True가 더 좋음
        optim="paged_adamw_8bit", # ★★★ 8비트 옵티마이저 사용 ★★★
        gradient_checkpointing=True, # ★★★ 그래디언트 체크포인팅 활성화 ★★★
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6) Trainer 생성 및 학습 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()