# Mutual-Prototype Adaptation for Cross-Domain Polyp Segmentation (JBHI 2021)


This is the official PyTorch implementation of **MPA** (Mutual-Prototype Adaptation) (JBHI 2021). 

**Mutual-Prototype Adaptation for Cross-Domain Polyp Segmentation**[\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9423517)

Chen Yang, Xiaoqing Guo, Meilu Zhu, Bulat Ibragimov, Yixuan Yuan

<div align="center">
  <img src="figs/framework.png"/>
</div>


Install dependencies
```
pip install -r requirements.txt
```

## Datasets Preparation

### CVC-DB and ETIS-Larib
(1) Download the [CVC-DB and ETIS-Larib](https://drive.google.com/drive/folders/1HqlgeYwqeh538lSmrAapCL2GP0zvUUH_?usp=sharing) dataset.

(2) Put the data in the corresponding folders.
The dataset files are organized as follows.
```
MPA-DA
├── data
│   ├── CVC-DB
│   │   ├── images
│   │   │   ├── [case_id].png
│   │   ├── labels
│   │   │   ├── [case_id].png
│   ├── ETIS-Larib
│   │   ├── images
│   │   │   ├── [case_id].png
│   │   ├── labels
│   │   │   ├── [case_id].png

```

(3) Split dataset into training set and test set as follows.

```
python preprocess.py
```
<!-- ## Data
You can download all datasets used in the paper from [here](https://1drv.ms/u/s!AtBnuAhBSAqjdjpLjYOq_geB1f4?e=zChYcN).  -->


## Quickstart
```
python ./tools/train.py
```

## Cite
If you find our work useful in your research or publication, please cite our work:
```
@article{yang2021mutual,
  title={Mutual-Prototype Adaptation for Cross-Domain Polyp Segmentation},
  author={Yang, Chen and Guo, Xiaoqing and Zhu, Meilu and Ibragimov, Bulat and Yuan, Yixuan},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2021},
  publisher={IEEE}
}
```
