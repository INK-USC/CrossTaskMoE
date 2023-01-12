## Eliciting and Understanding Cross-Task Skills with Task-Level Mixture-of-Experts 

This repository contains code accompanying our preprint paper "Eliciting and Understanding Cross-Task Skills with Task-Level Mixture-of-Experts" ([Paper](https://arxiv.org/pdf/2205.12701.pdf)).

In this paper, our goal is to better understand multi-task learning with mixture-of-experts models with an explicit routing mechanism. We also find that such design help improve the model's capability to generalize/adapt to new tasks in both few-shot setting and zero-shot setting. 

Our problem setting closely resembles [CrossFit](https://github.com/INK-USC/CrossFit) where you can find the way to build the dataset (NLP Few-shot Gym) we used for few-shot setting experiments in this repository.


:smiley: Please check `./example_scripts` for more experiment details!

### Training a task-level routing MOE 

Please refer to `./example_scripts/train_routing_bart.sh`.

Notes:
- The script will load the original bart-base weights by default, then copy and paste the parameters into *m* experts with extra random noise.
- In the script, you need to specify the `DATA_DIR` which is the folder for datasets. You also need to specify `SPLIT` which is a json file containing the sections for  train/dev/test tasks, referring to  [random.json](https://github.com/INK-USC/CrossFit/blob/master/dataloader/custom_tasks_splits/random.json). `TASKVECS` specifies the numpy array file that contains the initial values for all task vector.  The result model will be saved in directory `OUTPUT_DIR`.

### Training other MOE baselines
You can add  parameter `--avg_routing`/ `--task_level_random_routing`/`--random_routing` to `./example_scripts/train_routing_bart.sh` to explore Average  routing/Instance-level random/Task-level random routing MOE models mentioned in our paper section 5.2.

### Two-stage training

 we find that introducing routing mechanism naively may lead to worsened performance. Also, average routing is stable and achieves competitive performance. Based on these observations, we design a two-stage training strategy to combine the benefits of both methods. In the first stage, the model jointly learns the router and the experts (Training a task-level routing MOE), In the second stage, the experts are re-initialized from BART's pre-trained weights, and the routes gradually transforms from average routing to the learned routes by controlling the temperature used in the softmax function. 
 1. use `./example_scripts/build_a_checkpoint2.sh` to combine the learned routs in first stage with pre-trained bart weights into a single checkpoint.
 2. refer to `./example_scripts/run_merged_fisher128freeze_1e-5.sh` to retrain the model with gradually transferred routs.
***

### Useful Tools
- `./example_scripts/finetune_a_list_of_tasks.sh` will help you fine-tune a list of tasks sequentially, given a certain model initialization.
- `./example_scripts/collect_results.py` will read each `results.csv` files in a given directory, then compute mean and standard deviation of dev/test performance.
- `./task/task_emb` give examples to calculate different type of task embeddings.
***

### Contact Us
If you find bugs in our code, encounter problems when running the code, or have any suggestions, please submit an issue, or reach out to Qinyuan (qinyuany@usc.edu) and Juan Zha (juanzha@usc.edu)!

If you used our code in your study, or find our paper useful, please cite us using the BibTex below.

<details>
<summary>BibTeX</summary>

```
@article{ye2022eliciting,
  title={Eliciting Transferability in Multi-task Learning with Task-level Mixture-of-Experts},
  author={Ye, Qinyuan and Zha, Juan and Ren, Xiang},
  journal={arXiv preprint arXiv:2205.12701},
  year={2022}
}
```
</details>
