# SLOTAlign

This is a Python implementation of 

> Robust Attributed Graph Alignment via Joint Structure Learning and Optimal Transport
> 
> *Jianheng Tang, Weiqi Zhang, Jiajin Li, Kangfei Zhao, Fugee Tsung, Jia Li*
> 
> **ICDE 2023**  [Arxiv](https://arxiv.org/abs/2301.12721) [IEEE Library](https://www.computer.org/csdl/proceedings-article/icde/2023/222700b638/1PByqVpGgh2)


Dependencies
--------------------------------
- python 3.9
- cuda 11.3
- pytorch 1.11
- dgl 0.8
- pyg 2.0.4
- scikit-learn 1.0.2
- networkx 2.8.4
- argparse 1.4.0


Alignment on Douban Online-Offline
--------------------------------
```
python SLOTAlign.py --config config/douban.json
```


Alignment on ACM-DBLP
--------------------------------
```
python SLOTAlign.py --config config/dblp.json
```


Alignment over Inconsistent Structures
--------------------------------
```
python SLOTAlign.py --dataset cora --truncate True --edge_noise 0.5
```
- dataset - cora/citeseer/facebook/ppi
- edge_noise - floats between 0 and 1


Alignment over Inconsistent Features
--------------------------------
```
python SLOTAlign.py --dataset cora --noise_type 1 --feat_noise 0.5
```
- dataset - cora/citeseer/facebook/ppi
- noise_type - 1: permutation, 2: truncation, 3: compression,
- feat_noise - floats between 0 and 1


Alignment on the DBP15K dataset for knowledge graph entity alignment
--------------------------------------------------------------------
The dataset and LaBSE embedding files can be downloaded from [Google Drive](https://drive.google.com/file/d/1cP6CxVWsqa9ngOM4St1PzFv5NBF_jxBG/view?usp=sharing)
```
python run_DBP15K.py
```

If you use this package and find it useful, please cite our paper using the following BibTeX. Thanks! :)
```
@inproceedings{tang2023robust,
    author={Tang, Jianheng and Zhang, Weiqi and Li, Jiajin and Zhao, Kangfei and Tsung, Fugee and Li, Jia},
    booktitle = {2023 IEEE 39th International Conference on Data Engineering (ICDE)},
    title = {Robust Attributed Graph Alignment via Joint Structure Learning and Optimal Transport},
    pages = {1638-1651},
    doi = {10.1109/ICDE55515.2023.00129},
    url = {https://doi.ieeecomputersociety.org/10.1109/ICDE55515.2023.00129},
    year={2023}
}
```
