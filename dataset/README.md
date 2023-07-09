# Dataset download and preparation 

## 1. ImageNet

### Info
1000 Class, 1.2million image

Image resolution varies, e.g., fish folder, smallest 75 x 56, largest 4288 x 2848. On average is 469x387 pixel.

### Download
The offical download [sites](https://www.image-net.org/download.php) does not support downloading anymore.

Use [Academic Torrent](xhttps://academictorrents.com/browse.php?search=ImageNet), download the torrent file first, and then use [TransmissionBT](https://transmissionbt.com/download) to download the dataset with torrent file.

```ILSVRC2012_img_train.tar``` ~150GB

```ILSVR2012_img_val.tar``` 

Or use a subset, ImageNet-1k from hugggingface, requires tokens through ```huggingface-cli login```
```
from datasets import load_dataset
dset = load_dataset('imagenet-1k', split='train', streaming=True, use_auth_token=True)
```
### Preprocessing
The training images for imagenet are already in appropriate subfolders (like n07579787, n07880968). You need to get the validation groundtruth and move the validation images into appropriate subfolders. To do this, download ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar and use the following commands [reference](https://git-disl.github.io/GTDLBench/datasets/imagenet/#:~:text=Download%20Imagenet%2D12%20dataset%20from,classes%20and%201.2%20million%20images.):
```
# extract train data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
