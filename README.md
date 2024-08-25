# Architecture Evaluation Code

#### This code is the basis for which the memorisation and performance for the paper:
Training Data Memorization \& Performance for Large Language Model Architectures– Transformers vs. State Space Models

The folders correspond to each chapter of experiments within the paper. The original repositories (from the papers detailing the original codes and attacks) that were adapted for this research are also linked.

The primary evaluation ran between
[Pythia](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) and
[Mamba](https://huggingface.co/collections/state-spaces/transformers-compatible-mamba-65e7b40ab87e5297e45ae406) that were both streamed from HuggingFace.


## [Prefix Attacks](prefixattacks/)
Based on: [Carlini et. al's work](https://github.com/ftramer/LM_Memorization)


```
python main.py --N 1000 --batch-size 10 --model1 path/to/pythia-2.8b --model2 path/to/pythia-160m --corpus-path monology/pile-uncopyrighted
```
where model1 is the reference larger model and model 2 is the actual model being examined

Dataset for 
[The Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted) was streamed from HuggingFace.


## [LiveBench](performance/)
Based on: [livebench benchmark](https://github.com/livebench/livebench)
```
pip install torch packaging # These need to be installed prior to other dependencies.
pip install -e .
```

to generate model answers

```
python gen_model_answer.py --model-path /path/to/pythia-2.8b --model-id pythia-2.8b --dtype bfloat16 --bench-name live_bench
```

where bench-name can be all benches for live_bench, or specific bench category by live_bench/math or live_bench/math/AMPS_Hard


To score the model outputs:

```
python gen_ground_truth_judgment.py --bench-name live_bench
```

To show all the results:
```
python show_livebench_results.py
```



## [Membership Inference Attacks](mia/)
Based on: [Mink++ repo](https://github.com/zjysteven/mink-plus-plus/)

- `run.py` will run the Min-K% and Min-K%++ attack on the WikiMIA dataset, corresponding perturbed dataset and specified model.


``` 
python run_.py --model EleutherAI/pythia-160m --dataset WikiMIA_length32 --perturbed_dataset WikiMIA_length32_perturbed
```
- `run_neighbor.py` will run the Neighbor attack on the WikiMIA dataset with the specified model.

``` 
python run_neighbor.py --model EleutherAI/pythia-160m --dataset WikiMIA_length32
```

[WikiMIA dataset](https://huggingface.co/datasets/swj0419/WikiMIA)  was streamed from HuggingFace.