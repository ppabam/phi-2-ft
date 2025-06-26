# phi-2-ft

- https://chatgpt.com/share/685cbf24-cc5c-8008-a220-66073eeb1573

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