# CUT
Segmentation assisted U-shaped multi-scale transformer for crowd counting

# Eniviroment
timm==0.5.4<br />python >=3.6

pytorch >=1.4

opencv-python

scipy

h5py 

pillow


# Models

The result on SHA is reproducable on Google Colab, apart from the hyper-parameters stated in the paper, you also need to set seed=15 and start-val=120. A similar result will be appeared at epoch 171. Don't forget to install timm==0.5.4. The higher version will cause an error.

Regards to other datasets, the model needs to be trained using a GPU with at least 24GB memory. We provide pre-trained models for them here. 
