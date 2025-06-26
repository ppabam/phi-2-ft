# phi-2-ft

- https://chatgpt.com/share/685cbf24-cc5c-8008-a220-66073eeb1573
- https://g.co/gemini/share/585ffa361806

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
$ dm run python scripts/train.py --train_file ./data/train.jsonl --num_train_epochs 3 --per_device_train_batch_size 2

Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:09<00:00,  4.59s/it]
trainable params: 13,107,200 || all params: 2,792,791,040 || trainable%: 0.4693
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 95.74 examples/s]
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
  0%|                                                                                                                                                                                                  | 0/3 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
{'train_runtime': 2.5827, 'train_samples_per_second': 1.162, 'train_steps_per_second': 1.162, 'train_loss': 1.0423914591471355, 'epoch': 3.0}                                                                                
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.16it/s]
```

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