# CUT
Segmentation assisted U-shaped multi-scale transformer for crowd counting
![avatar](/model.png)
# Eniviroment
timm==0.5.4<br />
python < 3.10<br />
pytorch >=1.4<br />
opencv-python<br />
scipy==1.6.2<br />
h5py <br />
pillow<br />
tqdm<br />


# Models

SHA can be trained on Google Colab, apart from the hyper-parameters stated in the paper. We train a total of 500 epochs and start evalutaion after the 120th epoch. If you want a similiar result, try seed=15. Don't forget to install timm==0.5.4. The higher version will cause an error.

Regards to other datasets, the model needs to be trained using a GPU with at least 24GB memory. More epochs are needed for a model to converge. We provide pre-trained models for them here. <br />
[SHA](https://drive.google.com/file/d/1OyRo8eqfHTvoxxCPOImaUe3Ll_g5JnWO/view?usp=sharing)
[QNRF](https://drive.google.com/file/d/19T-YScQ6g7hMYFfvRIfWRblxlBiPvStJ/view?usp=sharing)
[JHU++](https://drive.google.com/file/d/16m1zM4TNZGUi0_TDWAiqQdQEmJ1pn_Nq/view?usp=sharing)

The pretrained backbone model is provided here: [PcPvT](https://drive.google.com/file/d/1sIpkV5fVEsQASfXaXxqzWrsc0MUQIvZq/view?usp=sharing) <br />
You can also download it from [Official](https://drive.google.com/file/d/1wsU9riWBiN22fyfsJCHDFhLyP2c_n8sk/view?usp=sharing)

# Results
![avatar](/result.jpg)

If you find our work useful, please cite our paper:
```
@inproceedings{cut,
title={Segmentation Assisted U-shaped Multi-scale Transformer for Crowd Counting},
author={Yifei Qian and Liangfei Zhang and Xiaopeng Hong and Carl Donovan and Ognjen Arandjelovic},
booktitle={2022 British Machine Vision Conference},
year={2022},
}
```
# Acknowledgement
```
@InProceedings{Rong_2021_WACV,
    author    = {Rong, Liangzi and Li, Chunping},
    title     = {Coarse- and Fine-Grained Attention Network With Background-Aware Loss for Crowd Density Map Estimation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {3675-3684}
}
```
```
@inproceedings{chu2021Twins,
	title={Twins: Revisiting the Design of Spatial Attention in Vision Transformers},
	author={Xiangxiang Chu and Zhi Tian and Yuqing Wang and Bo Zhang and Haibing Ren and Xiaolin Wei and Huaxia Xia and Chunhua Shen},
	booktitle={NeurIPS 2021},
	year={2021}
}
```
