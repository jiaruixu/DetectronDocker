# Detectron docker
## Build the image:

```
cd docker
docker build -t detectron .
```

## Use container-fn

Get lowerbound

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound \
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
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
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
