---
license: cc-by-4.0
dataset_info:
  features:
  - name: id
    dtype: string
  - name: type
    dtype: string
  - name: prompt
    dtype: string
  - name: completion
    dtype: string
  - name: annotation_1
    dtype: string
  - name: annotation_2
    dtype: string
  - name: agreement
    dtype: bool
  - name: final_label
    dtype: string
  splits:
  - name: gpt4
    num_bytes: 264757
    num_examples: 450
  - name: llama2new
    num_bytes: 523074
    num_examples: 450
  - name: llama2orig
    num_bytes: 476580
    num_examples: 450
  - name: mistralguard
    num_bytes: 309478
    num_examples: 450
  - name: mistralinstruct
    num_bytes: 377642
    num_examples: 450
  - name: prompts
    num_bytes: 47890
    num_examples: 450
  download_size: 930253
  dataset_size: 1999421
configs:
- config_name: default
  data_files:
  - split: gpt4
    path: data/gpt4-*
  - split: llama2new
    path: data/llama2new-*
  - split: llama2orig
    path: data/llama2orig-*
  - split: mistralguard
    path: data/mistralguard-*
  - split: mistralinstruct
    path: data/mistralinstruct-*
  - split: prompts
    path: data/prompts-*
task_categories:
- conversational
language:
- en
size_categories:
- 1K<n<10K
---
# XSTest Dataset for Testing Exaggerated Safety

Note, this is an upload of the data found [here](https://github.com/paul-rottger/exaggerated-safety) for easier research use.
All credit to the authors of the [paper](https://arxiv.org/abs/2308.01263)

The test prompts are subject to Creative Commons Attribution 4.0 International license. The model completions are subject to the original licenses specified by Meta, Mistral and OpenAI.

### Loading the dataset
Use the following:
```
from datasets import load_dataset
dataset = load_dataset("natolambert/xstest-v2-copy)
prompts = dataset["prompts"] # select the correct subset(s)
```

The dataset is a `DatasetDict` with the following structure:
```
DatasetDict({
    gpt4: Dataset({
        features: ['id', 'type', 'prompt', 'completion', 'annotation_1', 'annotation_2', 'agreement', 'final_label'],
        num_rows: 450
    })
    llama2new: Dataset({
        features: ['id', 'type', 'prompt', 'completion', 'annotation_1', 'annotation_2', 'agreement', 'final_label'],
        num_rows: 450
    })
    llama2orig: Dataset({
        features: ['id', 'type', 'prompt', 'completion', 'annotation_1', 'annotation_2', 'agreement', 'final_label'],
        num_rows: 450
    })
    mistralguard: Dataset({
        features: ['id', 'type', 'prompt', 'completion', 'annotation_1', 'annotation_2', 'agreement', 'final_label'],
        num_rows: 450
    })
    mistralinstruct: Dataset({
        features: ['id', 'type', 'prompt', 'completion', 'annotation_1', 'annotation_2', 'agreement', 'final_label'],
        num_rows: 450
    })

})
```

The mapping for prompts is as follows (from the original below)
* `id_v1` removed,
* `id_v2` -> `id`
* empty column completion
* focus to annotation_1 and note to annotation_2
* empty `final_label`
```
    prompts: Dataset({
        features: ['id_v1', 'id_v2', 'type', 'prompt', 'focus', 'note'],
        num_rows: 450
    })
```
Please cite their work:
```
@misc{röttger2023xstest,
      title={XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models}, 
      author={Paul Röttger and Hannah Rose Kirk and Bertie Vidgen and Giuseppe Attanasio and Federico Bianchi and Dirk Hovy},
      year={2023},
      eprint={2308.01263},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```