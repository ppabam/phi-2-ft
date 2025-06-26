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
$ pdm run python scripts/train.py --train_file ./data/train.jsonl --num_train_epochs 3 --per_device_train_batch_size 2

tokenizer_config.json: 7.34kB [00:00, 47.1MB/s]
vocab.json: 798kB [00:00, 42.8MB/s]
merges.txt: 456kB [00:00, 89.4MB/s]
tokenizer.json: 2.11MB [00:00, 180MB/s]
added_tokens.json: 1.08kB [00:00, 8.99MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 99.0/99.0 [00:00<00:00, 1.37MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 735/735 [00:00<00:00, 10.5MB/s]
model.safetensors.index.json: 35.7kB [00:00, 188MB/s]
model-00002-of-00002.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 564M/564M [00:19<00:00, 28.6MB/s]
model-00001-of-00002.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.00G/5.00G [01:35<00:00, 52.2MB/s]
Fetching 2 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [01:35<00:00, 47.94s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [01:00<00:00, 30.43s/it]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 1.02MB/s]
Generating train split: 3 examples [00:00, 66.99 examples/s]
Map:   0%|                                                                                                                                                                                      | 0/3 [00:00<?, ? examples/s]
Traceback (most recent call last):
  File "/home/tom/code/phi-2-ft/scripts/train.py", line 74, in <module>
    main()
  File "/home/tom/code/phi-2-ft/scripts/train.py", line 46, in main
    tokenized_dataset = dataset.map(tokenize_fn, batched=False)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/datasets/dataset_dict.py", line 944, in map
    dataset_dict[split] = dataset.map(
                          ^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 557, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 3079, in map
    for rank, done, content in Dataset._map_single(**dataset_kwargs):
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 3501, in _map_single
    for i, example in iter_outputs(shard_iterable):
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 3475, in iter_outputs
    yield i, apply_function(example, i, offset=offset)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 3398, in apply_function
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/scripts/train.py", line 42, in tokenize_fn
    tokens = tokenizer(prompt, max_length=args.max_seq_length, truncation=True, padding="max_length")
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 2867, in __call__
    encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 2977, in _call_one
    return self.encode_plus(
           ^^^^^^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 3043, in encode_plus
    padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
                                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tom/code/phi-2-ft/.venv/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 2769, in _get_padding_truncation_strategies
    raise ValueError(
ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
```