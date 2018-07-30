## Detectron docker ##
Build the image:

```
cd docker
docker build -t detectron .
```

Use container-fn

Get upperbound1

```
container-fn detectron-faster-rcnn-train \
      --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml \
      --output-path /mnt/fcav/self_training/object_detection/upperbound1 \
      --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

Get lowerbound

```
container-fn detectron-faster-rcnn-train \
      --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound.yaml \
      --output-path /mnt/fcav/self_training/object_detection/lowerbound \
      --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```
