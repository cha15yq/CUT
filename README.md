# CUT
Segmentation assisted U-shaped multi-scale transformer for crowd counting
![avatar](/model.png)
# Eniviroment
timm==0.5.4<br />
python >=3.6<br />
pytorch >=1.4<br />
opencv-python<br />
scipy<br />
h5py <br />
pillow<br />


# Models

The result on SHA is reproducable on Google Colab, apart from the hyper-parameters stated in the paper, you also need to set seed=15 and start-val=120. A similar result will be appeared at epoch 171. Don't forget to install timm==0.5.4. The higher version will cause an error.

Regards to other datasets, the model needs to be trained using a GPU with at least 24GB memory. We provide pre-trained models for them here. <br />
[SHA](https://drive.google.com/file/d/1OyRo8eqfHTvoxxCPOImaUe3Ll_g5JnWO/view?usp=sharing)
[QNRF](https://drive.google.com/file/d/19T-YScQ6g7hMYFfvRIfWRblxlBiPvStJ/view?usp=sharing)
[JHU++](https://drive.google.com/file/d/16m1zM4TNZGUi0_TDWAiqQdQEmJ1pn_Nq/view?usp=sharing)

The pretrained backbone model is provided here: [PcPvT](https://drive.google.com/file/d/1dmqaEGMJz-J8clV62YPS_XplNdUbXiis/view?usp=sharing) <br />
You can also download it from [Official](https://drive.google.com/file/d/1wsU9riWBiN22fyfsJCHDFhLyP2c_n8sk/view?usp=sharing)

# Results
![avatar](/result.jpg)
