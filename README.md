# GCN_SHS_NET
Suguru Honda, Koichiro Yano, Eiichi Tanaka, Katsunori Ikari, Masayoshi Harigai

The objective of this project is to validate whether graph neural networks and random forests can accurately predict the progression of joint destruction. The Sharp/van der Heijde score (SHS) is based on AI-predicted scores. [See also this repository.](https://github.com/honda-s691470/SHS_NET)

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
- boruta_py >= 0.3

# References
- The following papers and codes were the main references for the construction of the GCNs used in this study and for understanding the theory.  
Wang T, Shao W, Huang Z, et al. MOGONET integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification. Nat Commun 2021;12:3445. doi:10.1038/s41467-021-23774-w　[https://github.com/txWang/MOGONET](https://github.com/txWang/MOGONET)  
Copyright (c) 2021 txWang  
the source code of MOGONET is released under the MIT License  
[https://github.com/txWang/MOGONET/blob/main/LICENSE](https://github.com/txWang/MOGONET/blob/main/LICENSE)

- code of Boruta can be found [here](https://github.com/scikit-learn-contrib/boruta_py)   
Kursa MB, Rudnicki WR. Feature Selection with the Boruta Package. J Stat Softw 2010;36. doi:10.18637/jss.v036.i11  
Copyright (c) 2016, Daniel Homola  
the source code of Boruta is released under the BSD 3-Clause "New" or "Revised" License  
[https://github.com/scikit-learn-contrib/boruta_py/blob/master/LICENSE](https://github.com/scikit-learn-contrib/boruta_py/blob/master/LICENSE)
