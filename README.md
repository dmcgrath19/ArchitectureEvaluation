# Architecture Evaluation Code

#### This code is the basis for which the memorisation and performance for the paper:
Training Data Memorization \& Performance for Large Language Model Architectures– Transformers vs. State Space Models

The folders are correspond to each chapter of experiments within the paper, with the original repos(from the papers of original codes/attacks) that were adapted for this research linked as well.

## [Prefix Attacks](prefixattacks/)
Based on: [Carlini et. al's work](https://github.com/ftramer/LM_Memorization)


```
python main.py --N 1000 --batch-size 10 --model1 path/to/pythia-2.8b --model2 path/to/pythia-160m --corpus-path monology/pile-uncopyrighted
```
where model1 is the reference larger model and model 2 is the actual model being examined


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