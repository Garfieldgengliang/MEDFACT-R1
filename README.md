Code and datasets of paper "MEDFACT-R1: TOWARDS FACTUAL MEDICAL REASONING VIA PSEUDO-LABEL AUGMENTATION".

For GRPO code, please refer to src/r1-v/src/open_r1.

Dataset will be available soon.


# MEDFACT-R1: TOWARDS FACTUAL MEDICAL REASONING VIA PSEUDO-LABEL AUGMENTATION

*We integrate external knowledge grounding with reinforcement learning to improve the factual medical reasoning.


## 🌟 Requirements
1. Clone this repository
```bash

```

2. Install Package: Create conda environment

```Shell

```

3. Download the required model checkpoints [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) from huggingface.

4. For all the training datasets, you can download the dataset by links below.

- [MIMIC-CXR](https://drive.google.com/file/d/1Yd4MVqbC9eegMOWORRjhW-kqgcR2pzoH/view?usp=drive_link)
- [IU-Xray](https://drive.google.com/file/d/1q9VTk8OW-H2TLlbPCcN-nrY5xVI3iMns/view?usp=sharing) 
- [Harvard-FairVLMed](https://drive.google.com/file/d/1czgGimDWmfS1cRnNsbgPEzQgA900ia-e/view?usp=sharing)

## 📖 Data Description
We provide a corresponding json file for generating each dataset.

- Training: Download the dataset from google drive, and place them under sepcific directory. Then use python scripts from `/data_generate` to generate Datasets file for RL training.
- Test: All the test data for test inference MEDFACT-R1 is placed under `/inference`. 

## 🚀 GRPO Training

- The example training script is at `r1-v/run_grpo_vllm_lgl_0526.sh`.


## 📚 Citation


## 🙏 Acknowledgement
We use code from [R1-V](https://github.com/StarsfieldAI/R1-V). We thank the authors for releasing their code.

