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
这里是一个小的Demo，经过这个demo，您可以绘制出文章中的calibration curves。
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
#### 2. Training and get results
这里以ensemble-based的模型为例。在保证环境与Installation中描述的一致后，找到./src/UQ_Ensemble.py，该脚本可以直接运行，您也可以对其函数参数进行配置。如需运行：
```bash
python ./src/UQ_Ensemble.py
```
训练结束后，模型的训练参数会保存在./checkpoints文件夹下，实验中用到的结果会保存在./result下。
### 3. Plotting
在得到结果后，可以运行./src/evaluation.py进行绘制图像。注意CaliCurvePlot函数的参数需要和训练模型时的保持一致。
```bash
python ./src/evaluation.py
```
