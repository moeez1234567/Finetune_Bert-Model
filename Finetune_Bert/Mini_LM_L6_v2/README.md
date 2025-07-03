---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:11247
- loss:CosineSimilarityLoss
- dataset_size:2873
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: “You are so good,” he said.
  sentences:
  - 'I waved my hand to her, and nodded to

    tell that our work there was successfully accomplished.'
  - If you will let me, I shall give you a paper to read.
  - '“And may

    I read it now?'
- source_sentence: 'I wonder if

    Renfield’s quiet has anything to do with this.'
  sentences:
  - 'His moods have so

    followed the doings of the Count, that the coming destruction of the

    monster may be carried to him in some subtle way.'
  - 'As

    his room is on this side of the house, I could hear it better than in

    the morning.'
  - '(_Mem._, I must ask the Count about these

    superstitions)


    When we started, the crowd round the inn door, which had by this time

    swelled to a considerable size, all made the sign of the cross and

    pointed two fingers towards me.'
- source_sentence: 'I am in hopes that we need have no inquest, for if we

    had it would surely kill poor Lucy, if nothing else did.'
  sentences:
  - 'He stopped to talk with me, as he always does, but all the time

    kept looking at a strange ship.'
  - 'After a pause

    of a moment, he proceeded, in his stately way, to the door, drew back

    the ponderous bolts, unhooked the heavy chains, and began to draw it

    open.'
  - 'He came back with a handful of wild garlic

    from the box waiting in the hall, but which had not been opened, and

    placed the flowers amongst the others on and around the bed.'
- source_sentence: 'How he has been

    making use of the zoöphagous patient to effect his entry into friend

    John’s home; for your Vampire, though in all afterwards he can come when

    and how he will, must at the first make entry only when asked thereto by

    an inmate.'
  sentences:
  - 'Van Helsing is off to the

    British Museum looking up some authorities on ancient medicine.'
  - 'Too well I know the broad

    fact; tell me all that has been.” I told him exactly what had happened,

    and he listened with seeming impassiveness; but his nostrils twitched

    and his eyes blazed as I told how the ruthless hands of the Count had

    held his wife in that terrible and horrid position, with her mouth to

    the open wound in his breast.'
  - But these are not his most important experiments.
- source_sentence: Yes!
  sentences:
  - 'Were you not amazed, nay

    horrified, when I would not let Arthur kiss his love--though she was

    dying--and snatched him away by all my strength?'
  - 'Henceforth our work is to be a

    sealed book to her, till at least such time as we can tell her that all

    is finished, and the earth free from a monster of the nether world.'
  - They will be home by this.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Yes!',
    'Were you not amazed, nay\nhorrified, when I would not let Arthur kiss his love--though she was\ndying--and snatched him away by all my strength?',
    'They will be home by this.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.3134,  0.4086],
#         [ 0.3134,  1.0000, -0.0246],
#         [ 0.4086, -0.0246,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2,873 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                          |
  | details | <ul><li>min: 4 tokens</li><li>mean: 30.58 tokens</li><li>max: 197 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 30.29 tokens</li><li>max: 172 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.33</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                             | sentence_1                                                                                                                        | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>I<br>may not have the pleasure to talk to-night, since there are many labours<br>to me; but you will sleep, I pray.” I passed to my room and went to bed,<br>and, strange to say, slept without dreaming.</code> | <code>Despair has its own calms.</code>                                                                                           | <code>1.0</code> |
  | <code>“I have an idea.</code>                                                                                                                                                                                          | <code>Once the flame appeared so near the road, that even in the darkness<br>around us I could watch the driver’s motions.</code> | <code>0.0</code> |
  | <code>We must have<br>another transfusion of blood, and that soon, or that poor girl’s life<br>won’t be worth an hour’s purchase.</code>                                                                               | <code>All correct.</code>                                                                                                         | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.7112 | 500  | 0.1645        |


### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 5.0.0
- Transformers: 4.41.0
- PyTorch: 2.7.1+cpu
- Accelerate: 0.29.0
- Datasets: 3.6.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->