# SCAN-Learn
This is an experiment of compositional learning and zero-shot generalization on the [SCAN task](https://github.com/brendenlake/SCAN) in [Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks](https://arxiv.org/abs/1711.00350) by *Brenden Lake and Marco Baroni*. The SCAN tasks were inspired by the CommAI environment, which is the origin of the acronym (Simplified versions of the CommAI Navigation tasks).  

## Requirements
- python 3.5+
- pytorch 0.3+

## Usage
1. Download data in the [SCAN task](https://github.com/brendenlake/SCAN);
2. Process the data with **data_process\*.py**;
3. Run the model with **\*s2s_[model].py** `[options]`.
