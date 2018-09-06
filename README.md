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
  --output-path /mnt/fcav/self_training/object_detection/upperbound1 \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

Get baseline

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

Get baseline with soft sampling

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss_0.99_ftl_add_images \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_trainconv.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss_trainconv \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn_and_backbone.pkl
```
Get upperbound2

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound2.yaml \
  --output-path /mnt/fcav/self_training/object_detection/upperbound2 \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

Test localization error

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test.yaml \
  --output-path /mnt/fcav/self_training/object_detection/localization_error_test \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_no_fn.yaml \
  --output-path /mnt/fcav/self_training/object_detection/localization_error_test_no_fn \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_localization_error_test_no_fn.yaml \
  --output-path /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_no_fn_with_10fp.yaml \
  --output-path /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss_with_10fp \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_no_fn_with_20fp.yaml \
  --output-path /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss_with_20fp \
  --region-proposal-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
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
baseline

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_GTA8k_pred_evalKITTIval1k.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_retrain_GTA8k_pred/eval \
  --test-weights /mnt/fcav/self_training/object_detection/baseline_retrain_GTA8k_pred/train/voc_GTA_caronly_train_sample8000_coco_KITTI_train_with_prediction/generalized_rcnn
```

baseline_ss

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_GTA8k_pred_evalKITTIval1k.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss_retrain_GTA8k_pred/eval \
  --test-weights /mnt/fcav/self_training/object_detection/baseline_ss_retrain_GTA8k_pred/train/voc_GTA_caronly_train_sample8000_coco_KITTI_train_with_prediction/generalized_rcnn
```

lowerbound

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound/eval \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn
```

### detectron-faster-rcnn-feedforward (generate region proposals or prediction)

#### Region proposals

on GTA 8k

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_GTA_train8k.yaml \
  --output-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

on KITTI prediction

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_train_prediction.yaml \
  --output-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

on KITTI val 1000
```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_eval_1k.yaml \
  --output-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

on KITTI train

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_train.yaml \
  --output-path /mnt/fcav/self_training/object_detection/region_proposals_GTA200k \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

#### Predictions

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/prediction_final \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```
