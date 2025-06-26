# phi-2-ft

- https://chatgpt.com/share/685cbf24-cc5c-8008-a220-66073eeb1573
- https://g.co/gemini/share/585ffa361806
- [중·노년층 한국어 방언 데이터(강원도, 경상도)](https://aihub.or.kr/aihubdata/data/view.do?searchKeyword=%EB%B0%A9%EC%96%B8&aihubDataSe=data&dataSetSn=71517)

### Folde structure
```bash
my-llm-project/
├── data/                  # 학습/평가용 데이터셋
│   ├── train.jsonl
│   └── valid.jsonl
├── scripts/               # 실행 스크립트
│   └── train.py
│   └── inference.py
├── models/                # 체크포인트 저장
├── notebooks/             # 시각화 또는 실험
├── pyproject.toml         # pdm 프로젝트 설정
├── README.md
└── .env                   # HuggingFace token 등
```

### Training
```
$ pdm run python scripts/train.py --train_file ./data/train.jsonl --num_train_epochs 3 --per_device_train_batch_size 2

Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:09<00:00,  4.59s/it]
trainable params: 13,107,200 || all params: 2,792,791,040 || trainable%: 0.4693
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 95.74 examples/s]
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
  0%|                                                                                                                                                                                                  | 0/3 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
{'train_runtime': 2.5827, 'train_samples_per_second': 1.162, 'train_steps_per_second': 1.162, 'train_loss': 1.0423914591471355, 'epoch': 3.0}                                                                                
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.16it/s]
```

### Training Details (QLoRA)

`scripts/train.py`는 QLoRA (Quantized Low-Rank Adaptation)를 사용하여 `microsoft/phi-2` 모델을 미세 조정합니다. 주요 특징은 다음과 같습니다:

*   **4-bit 양자화:** `BitsAndBytesConfig`를 통해 모델을 4-bit로 양자화하여 로드하여 메모리 사용량을 크게 줄입니다.
*   **NF4 (NormalFloat 4) 데이터 타입:** 4-bit 양자화에 최적화된 NF4 데이터 타입을 사용합니다.
*   **페이지드 옵티마이저 (Paged Optimizer):** `optim="paged_adamw_8bit"` 설정을 통해 CPU와 GPU 간의 메모리 페이징을 사용하여 메모리 부족 문제를 방지합니다.
*   **그래디언트 누적 (Gradient Accumulation):** `gradient_accumulation_steps`를 사용하여 여러 배치에 대한 그래디언트를 누적한 후 한 번에 모델 파라미터를 업데이트합니다.
*   **그래디언트 체크포인팅 (Gradient Checkpointing):** `gradient_checkpointing=True`를 활성화하여 학습 중 메모리 사용량을 줄입니다.

이러한 설정들은 제한된 GPU 메모리 환경에서도 효율적으로 대규모 언어 모델(LLM)을 미세 조정할 수 있도록 돕습니다.


### Interface
```bash
$ pdm run python scripts/interface_phi2.py --prompt "이것은 질문입니다"            
'microsoft/phi-2' 원본 모델을 로드합니다...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.06s/it]
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

==================================================
   원본 모델 [microsoft/phi-2]의 응답
==================================================

[입력된 전체 프롬프트]:
### 질문: 이것은 질문입니다
### 답변:

--------------------------------------------------

[생성된 답변]:
 이것은 질문입니다
### 전체 개발보호: 해당 코드는 바이러스에서 질문을 추출하는 가장 짧은 바이러스
### 파이썬 객체: 문자열
### 방문정보: 첫째 이용자
### 현재 공간: 그래�

==================================================

$ pdm run python scripts/interface.py --prompt "이것은 질문입니다"
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.21s/it]
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
==============================
질문: 이것은 질문입니다
생성된 답변:  이것은 답변입니다
### 다른 질문: 이것은 다른 질문입니다
### 다른 답변: 이것은 답변입니다

==============================
```

```bash
$ pdm run python scripts/interface_phi2.py --prompt "대한민국의 수도는 어디인가요?"
'microsoft/phi-2' 원본 모델을 로드합니다...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.99s/it]
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

==================================================
   원본 모델 [microsoft/phi-2]의 응답
==================================================

[입력된 전체 프롬프트]:
### 질문: 대한민국의 수도는 어디인가요?
### 답변:

--------------------------------------------------

[생성된 답변]:
 열러브는 어디를 만들어 지내기 위해서는 단어를 만들어 어디를 만들어 지내기 위해서는 단어를 만들어 어디를 만들어 지내기 위해서는 단어를 만들어 어디를 만들어 �

==================================================

$ pdm run python scripts/interface.py --prompt "대한민국의 수도는 어디인가요?"
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.07s/it]
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
==============================
질문: 대한민국의 수도는 어디인가요?
생성된 답변: 대한민국은 여기서 대한민국에서 시작하는 대한민국을 인식할 수 있는지?
==============================
```

## SATURI
* 표준어-방언 쌍 추출: JSON 파일 내의 transcription.segments 와 transcription.sentences 필드에서 dialect와 standard 키를 이용하여 추출 가능합니다.
* 지역별 구분: speaker.residenceProvince ("gs" - 경상도) 와 speaker.residenceCity (20.0) 필드를 통해 지역별 구분이 가능합니다.

```bash
$ pdm run python scripts/train.py --train_file ./data/train_andong_cleaned.jsonl --num_train_epochs 3 --per_device_train_batch_size 2
```