# Architecture Evaluation code

#### This code is the basis for which the memorisation and performance for the paper:
Training Data Memorization \& Performance for Large Language Model Architectures– Transformers vs. State Space Models

The folders are correspond to each chapter of experiments within the paper, with the original repos that helped guide the research(that were made public alongside their paper publication) linked as well.

## [Prefix Attacks](prefixattacks/)
Original Repo: [here](https://github.com/ftramer/LM_Memorization)

first install dependencies

```pip install -r requirements.txt```

example job script here:

```python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted```


## [LiveBench](performance/)
Original Repo: [here](https://github.com/livebench/livebench)
 

## [Membership Inference Attacks](mia/)
Original Repo [here](https://github.com/zjysteven/mink-plus-plus/)
