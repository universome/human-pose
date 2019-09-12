This repo contains some experiments with DensePose estimation

### How to run

1\. Download COCO images
```
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip
rm train2014.zip val2014.zip
cd ../..

```
2\. Download DensePose annotations
```
mkdir -p data/densepose && cd data/densepose
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_valminusminival.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_minival.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_test.json
cd ../..
```

3\. If you want to run training, you should also download some evaluation data
```
mkdir -p data/densepose/eval_data && cd data/densepose/eval_data 
wget https://dl.fbaipublicfiles.com/densepose/densepose_eval_data.tar.gz
tar xvf densepose_eval_data.tar.gz
rm densepose_eval_data.tar.gz
cd ../../..
```

4\. Next, you should install the libraries
```
pip install -r requirements.txt
```

5\. Finally you can start training by running:
```
python run_trainer.py --experiments_dir=my-experiments
```
Experiment results will be saved into `my-experiments` directory.
If you want to run on GPU you should additionally provide `--available_gpus` argument specifying which GPUs you allow to use.

### Ideas/TODOs
* Even in MaskTrack R-CNN we run backbone for each frame independently. Can we train some kind of recurrent model (in an unsupervised way), which we'll try to predict embeddings, which are predicted by backbone? It should be much cheaper, isn't it?.
* How can MaskTrack R-CNN run 20 FPS if it is identical to normal Mask-RCNN, but with a new tracking head?
* Can we compare in MaskTrack R-CNN not to the last object feature, but to the history of them (or running mean)? It should be more robust. But this does not seem necessary, because Identity Oracle performs not much better.
* Why don't we use skip-connections in Mask R-CNN like we do in U-net? They should help model to predict the mask.
* Can we replace RoIAlign with attention? For example, we can predict attention weights instead of predicting bboxes for each sliding window over features.
* What if we train object detector to predict from bottom to top, from right to len?
* Maybe it will be easer for model to predict angle/length bbox adjustments instead of dx/dy adjustments?
* Maybe we'll be fine with single-stage detectors since we only care about catching humans (and, besides that, of humans of a quite large size, i.e. those, who are close to a camera)
* Points which lie on borders between two different parts are hard to distinguish for the model. And not because our model is stupid, but because our parametrization is stupid. And there are quite a lot of such borders. Use different parametrization.
* It feels that we can interpolate DensePose targets for additional supervision.
* Use SNIPER approach for large image and combine results together via postprocessing: this will allow us to process patches in a batch (should speed things up)
* Maybe it's better to upsample output intstead of downsampling targets?
* Maybe DeepLabV3 didn't work for authors because they didn't care enough about class imbalance (use something like Focal Loss)?
* Maybe our bbox head should be of size Kx4, where K is number of classes? This can help the model to differentiate. But actually we do almost the same by using different anchor sizes, i.e. outputting Ax4.
* Maybe we should also initialize DP heads properly?
* Total variation regularizer
