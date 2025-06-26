import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def parse_args():
    """커맨드 라인 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description="Interface with the original phi-2 model")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2", help="Base model name or path")
    parser.add_argument("--prompt", type=str, required=True, help="The core question to ask the model")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"'{args.model_name}' 원본 모델을 로드합니다...")

    # 1. 모델을 4-bit로 로드하기 위한 설정 (메모리 절약)
    # fine-tuned 모델 추론과 동일한 조건으로 비교하기 위해 양자화를 적용합니다.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2. 원본 모델과 토크나이저 로드 (LoRA 어댑터 없음)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # 현재 GPU에 모델 할당
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()  # 추론 모드로 설정

    # 3. 파인튜닝 모델과 동일한 형식의 프롬프트 준비
    # 공정한 비교를 위해, 파인튜닝된 모델이 입력받는 것과 완전히 동일한 형태로 프롬프트를 구성합니다.
    formatted_prompt = f"### 질문: {args.prompt}\n### 답변:"

    # 4. 토큰화 및 추론
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # 원본 모델은 더 길게 답할 수 있으므로 넉넉하게 설정
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 5. 생성된 답변 부분만 디코딩
    response_tokens = outputs[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # 6. 결과 출력
    print("\n" + "="*50)
    print(f"   원본 모델 [microsoft/phi-2]의 응답")
    print("="*50)
    print(f"\n[입력된 전체 프롬프트]:\n{formatted_prompt}")
    print("\n" + "-"*50)
    print(f"\n[생성된 답변]:\n{response}")
    print("\n" + "="*50)


if __name__ == "__main__":
    main()