# Dataset download and preparation 

## 1. ImageNet (ILSVRC2012)

### Info
This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images.

Image resolution varies, e.g., fish folder, smallest 75 x 56, largest 4288 x 2848. On average is 469x387 pixel.

### Download
1. The offical download [sites](https://www.image-net.org/download.php) does not support downloading data anymore. Only the toolkits works.
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
```

3. Use [Academic Torrent](https://academictorrents.com/browse.php?search=ImageNet), download the torrent file first, and then use [TransmissionBT](https://transmissionbt.com/download) to download the dataset with torrent file.

```
# download the torrent file
wget https://academictorrents.com/download/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent
wget https://academictorrents.com/download/a306397ccf9c2ead27155983c254227c0fd938e2.torrent
```
```
apt-get install transmission-cli
transmission-cli a306397ccf9c2ead27155983c254227c0fd938e2.torrent -w ~/Downloads
transmission-cli 5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent -w ~/Downloads
```
- ```ILSVRC2012_img_train.tar``` 147.9GB

- ```ILSVR2012_img_val.tar``` 6.74GB

3. Or use [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) from hugggingface, requires tokens through ```huggingface-cli login```
```
from datasets import load_dataset
dset = load_dataset('imagenet-1k', split='train', streaming=True, use_auth_token=True)
```

4. Or download from Kaggle, [object localization competition](kaggle competitions download -c imagenet-object-localization-challenge)

```
pip install kaggle
#put your kaggle API token, in /root/.kaggle/kaggle.json file, you can download your json from Kaggle account
kaggle competitions download -c imagenet-object-localization-challenge #register the competition first
```
[Reference example](https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/)

5. Alternative, use Tiny-ImageNet, Tiny ImageNet contains 100000 images of 200 classes (500 for each class) downsized to 64×64 colored images. Each class has 500 training images, 50 validation images and 50 test images.
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
[Reference example](https://rocm.docs.amd.com/en/latest/examples/machine_learning/pytorch_inception.html)

### Preprocessing
- The training images for imagenet are already in appropriate subfolders (like n07579787, n07880968). You need to get the validation groundtruth and move the validation images into appropriate subfolders. To do this, download ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar and use the following commands [reference](https://git-disl.github.io/GTDLBench/datasets/imagenet/#:~:text=Download%20Imagenet%2D12%20dataset%20from,classes%20and%201.2%20million%20images.):
```
# extract train data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
- or refer to pytorch imageNet [example](https://github.com/pytorch/examples/tree/main/imagenet)
