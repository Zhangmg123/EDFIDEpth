
This is the official PyTorch implementation code for EDFIDepth. For technical details, please refer to:

## Contents
1. [Datasets](#datasets)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Models](#models)
5. [Demo](#demo)


## Datasets
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/cleinc/bts), and then modify the data path in the config files to your dataset locations.

Or you can download the NYUv2 data from [here](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/datasets/nyu/sync.zip) and download the KITTI data from [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).


## Training
First download the pretrained encoder backbone from [here](https://github.com/youngwanLEE/MPViT), and then modify the pretrain path in the config files.

Training the NYUv2 model:
```
python EDFIDepth/train.py configs/arguments_train_nyu.txt
```

Training the KITTI model:
```
python EDFIDepth/train.py configs/arguments_train_kittieigen.txt
```


## Evaluation
Evaluate the NYUv2 model:
```
python EDFIDepth/eval.py configs/arguments_eval_nyu.txt
```

Evaluate the KITTI model:
```
python EDFIDepth/eval.py configs/arguments_eval_kittieigen.txt
```


## Demo
Test images with the indoor model:
```
python EDFIDepth/test.py --data_path datasets/test_data --dataset nyu --filenames_file data_splits/test_list.txt --checkpoint_path model_nyu.ckpt --max_depth 10 --save_viz
```

## Acknowledgements
Thanks to Jin Han Lee for opening source of the excellent work [BTS](https://github.com/cleinc/bts).
Thanks to Microsoft Research Asia for opening source of the excellent work [MPViT](https://github.com/youngwanLEE/MPViT).
