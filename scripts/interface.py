import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with a fine-tuned LoRA model")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-2", help="Base model name or path")
    parser.add_argument("--lora_adapter", type=str, default="./models/qlora-phi2", help="Path to the trained LoRA adapter")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate a response for")
    return parser.parse_args()

def main():
    args = parse_args()

    # ... (이전과 동일한 모델 로드 부분) ...
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    model.eval()
    
    formatted_prompt = f"### 질문: {args.prompt}\n### 답변:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # 5. 결과 디코딩 및 출력
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 수정된 핵심 로직: \n### 기준으로 자르기 ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    full_response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    # 생성된 텍스트에서 다음 질문 패턴("\n###")이 시작되는 부분을 기준으로 잘라냅니다.
    response = full_response.split("\n###")[0].strip()
    
    print("="*30)
    print(f"질문: {args.prompt}")
    print(f"생성된 답변: {response}")
    print("="*30)

if __name__ == "__main__":
    main()