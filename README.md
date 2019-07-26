This repo contains some experiments with DensePose estimation

TODO:
* Implement Mask-RCNN
* Implement DensePose base on Mask-RCNN
* Implement LSTM-based DensePose for video

Ideas:
* Even in MaskTrack R-CNN we run backbone for each frame independently. Can we train some kind of recurrent model (in an unsupervised way), which we'll try to predict embeddings, which are predicted by backbone? It should be much cheaper, isn't it?.
* How can MaskTrack R-CNN run 20 FPS if it is identical to normal Mask-RCNN, but with a new tracking head?
* Can we compare in MaskTrack R-CNN not to the last object feature, but to the history of them (or running mean)? It should be more robust. But this does not seem necessary, because Identity Oracle performs not much better.
* Why don't we use skip-connections in Mask R-CNN like we do in U-net? They should help model to predict the mask.
* Can we replace RoIAlign with attention? For example, we can predict attention weights instead of predicting bboxes for each sliding window over features.
* What if we train object detector to predict from bottom to top, from right to len?
* Maybe it will be easer for model to predict angle/length bbox adjustments instead of dx/dy adjustments?
* Maybe we'll be fine with single-stage detectors since we only care about catching humans (and, besides that, of humans of a quite large size, i.e. those, who are close to a camera)
* Points which lie on borders between two different parts are hard to distinguish for the model. And not because our model is stupid, but because our parametrization is stupid. And there are quite a lot of such borders. Use different parametrization.
* It feels like we can interpolate DensePose targets for additional supervision.
