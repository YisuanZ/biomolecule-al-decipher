# [Biomolecule-AL-Decipher] Deciphering Key Factors of Active LearningPerformance in Biomolecular Design

This repository hosts the official code and datasets for the research paper *Deciphering Key Factors of Active Learning Performance in Biomolecular Design*.

### Core Focus
We systematically investigate the key factors of Active Learning (AL) performance in biomolecular design, and provide a comprehensive benchmark for uncertainty quantification (UQ) algorithms and AL sampling strategies. Our findings reveal the critical factors influencing AL-based fitness optimization efficiency, offering practical guidance for genetic engineering applications.

### Key Outcomes
- Demonstrated the reliability and robustness of ensemble-based UQ algorithms
- Evaluated representative existing AL sampling strategies 
- Identified significant factors determining optimization efficiency (initial settings, distribution sparsity, and sequence similarity)
- Proposed two quantifiable metrics to measure the key influencing factors

### Quick Start
#### 1. Installation
First, clone the repository to your local machine:
```bash
git clone https://github.com/YisuanZ/biomolecule-al-decipher.git
cd biomolecule-al-decipher
```
Then, set up the corresponding virtual environment according to the experiment type. Note that the separation of environments below is due to the special configuration requirements of our laboratory server. For experiment with ensemble-based and MC dropout UQ algorithms:
```bash
# Create and activate Conda virtual environment (named as you like)
conda create -n AL python=3.9.6
conda activate AL
# Install all required dependencies in the environment
pip install -r requirements.txt
```
For experiment with DKL UQ algorithm:
```bash
# Create and activate Conda virtual environment (named as you like)
conda create -n ALdkl python=3.9.6
conda activate ALdkl
# Install all required dependencies in the environment
pip install -r requirements_dkl.txt
```
