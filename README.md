# GCN_SHS_NET
Suguru Honda, MD, Koichiro Yano, MD, PhD, Eiichi Tanaka, MD, PhD, Katsunori Ikari, MD, PhD, Masayoshi Harigai, MD, PhD

This project is validating whether graph neural networks and random forests can accurately predict the progression of joint destruction. The Sharp/van der Heijde score (SHS) is based on AI-predicted scores. [See also this repository.](https://github.com/honda-s691470/SHS_NET)

# Code Architecture
<pre>
.　　                
├── RF_weight_log       
│   └── SHS_NET_wrist_erosion             # Directory to store config, log, results of visualization and weight parameter files in analysis of Random Forest             
├── SHS_AI_wristE_s0.1_d0.6               # Directory to store config, log, summary of statistics, and weight parameter files in analysis of Graph neural network
├── GCN_SHS_AI_with_featimp_wristE.ipynb  # main code for graph neural network
├── LICENSE                               # LICENSE file
├── README.md                             # README file 
├── RF_for_SHS_NET_Eosion_wrist.ipynb     # main code for random forest with Boruta
├── models.py                             # code for architecuture of graph neural network 
├── train_test.py                         # code for training and testing 
├── utils.py                              # common useful modules (to make scheduler, cosine similarity, adjacent matrix etc.)    
└── wrist_erosion_1503.csv                # csv file including image id and variables   
</pre> 


# Recommended Requirements
This code was tested primarily on Python 3.8.12 using jupyter notebook.
The following environment is recommended.

for GCN
- pytorch　>= 1.7.1
- Numpy >= 1.21.4
- Pandas >= 1.13
- Matplotlib >= 3.3.1
- Seaborn >= 0.11.0
- Sklearn >= 0.23.2

for RF
- Sklearn >= 0.23.2
- dtreeviz >= 1.3.1
- graphviz >= 2.38

This repository contains dummy images obtained from [RSNA-Pediatric-Bone-Age-Challenge-2017](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017)   
Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al. The RSNA Pediatric Bone Age Machine Learning Challenge. Radiology 2018; 290(2):498-503.

code of adabound can be found [here](https://github.com/Luolc/AdaBound)  
Luo L, Xiong Y, Liu Y, et al. Adaptive Gradient Methods with Dynamic Bound of Learning Rate. Published Online First: 26 February 2019.http://arxiv.org/abs/1902.09843
