# Detectron docker
## Build the image:

```
cd docker
docker build -t detectron .
```

## Use container-fn

### detectron-faster-rcnn-train

Get lowerbound

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_200k.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound_200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

Get upperbound1

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound1.yaml \
  --output-path /mnt/fcav/self_training/object_detection/upperbound1_24_27 \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

Get baseline

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTAPretrained/model_wo_fast_rcnn.pkl
```

Get baseline with soft sampling

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTAPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_trainconv.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss_trainconv \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTAPretrained/model_wo_fast_rcnn_and_backbone.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_GTAonly.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss_GTAonly \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTAPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_trainall.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss_trainall \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTAPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss_lr \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTAPretrained/model_wo_fast_rcnn.pkl
```

Get upperbound2

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound2.yaml \
  --output-path /mnt/fcav/self_training/object_detection/upperbound2 \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

Test soft sampling

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_all_annotation_test.yaml \
  --output-path /mnt/fcav/self_training/object_detection/soft_sampling_test/all_annotations \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_drop30test.yaml \
  --output-path /mnt/fcav/self_training/object_detection/soft_sampling_test/drop30 \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

### detectron-faster-rcnn-eval

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound/eval \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn
```
