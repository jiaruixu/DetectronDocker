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
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound1.yaml \
  --output-path /mnt/fcav/self_training/final_results/upperbound1 \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound1_cityscapes.yaml \
  --output-path /mnt/fcav/self_training/final_results/cityscapes/upperbound \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

Get baseline

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_not_freeze.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_not_freeze \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_cityscapes.yaml \
  --output-path /mnt/fcav/self_training/final_results/cityscapes/baseline_not_freeze \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_f_not_freeze.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_ss_f_not_freeze \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_f_not_freeze_2nd_boot.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_f_not_freeze_2nd_boot \
  --pretrained-weights /mnt/fcav/self_training/final_results/baseline_f_not_freeze/train/model_iter8999.pkl
```

Get baseline with soft sampling

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_ss \
  --region-proposal-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_f.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_ss_f \
  --region-proposal-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/GTA200kPretrained/model_wo_fast_rcnn.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_f_not_freeze.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_ss_f_not_freeze2 \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_f_not_freeze_2nd_boot_union.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_2nd_boot_union \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_f_not_freeze_2nd_boot_selected_ss.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_2nd_boot_selected_ss \
  --pretrained-weights /mnt/fcav/self_training/final_results/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
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
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_localization_error_test.yaml \
  --output-path /mnt/fcav/self_training/final_results/localization_error \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_localization_error_test_no_fn.yaml \
  --output-path /mnt/fcav/self_training/final_results/localization_error_no_fn \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_localization_error_test_no_fn_10fp.yaml \
  --output-path /mnt/fcav/self_training/final_results/localization_error_no_fn_with_10fp \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_localization_error_test_no_fn_25fp.yaml \
  --output-path /mnt/fcav/self_training/final_results/localization_error_no_fn_with_25fp \
  --pretrained-weights /mnt/fcav/self_training/object_detection/pretrained_model/ImageNetPretrained/X-101-64x4d.pkl
```

```
container-fn detectron-faster-rcnn-train \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_localization_error_test_no_fn_20fp.yaml \
  --output-path /mnt/fcav/self_training/final_results/localization_error_no_fn_with_20fp \
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

lowerbound

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound/eval \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn
```

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
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_ss/eval \
  --test-weights /mnt/fcav/self_training/final_results/baseline_ss/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_train_with_prediction/generalized_rcnn
```

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_ss_f_less_iter/eval \
  --test-weights /mnt/fcav/self_training/final_results/baseline_ss_f_less_iter/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_train_with_prediction_ftl/generalized_rcnn
```

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_ss_f_lower_lr2/eval \
  --test-weights /mnt/fcav/self_training/final_results/baseline_ss_f_lower_lr2/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_train_with_prediction_ftl/generalized_rcnn
```

upperbound1

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/upperbound1/eval \
  --test-weights /mnt/fcav/self_training/final_results/upperbound1/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_train/generalized_rcnn
```

localization errors

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/localization_error_no_fn_with_20fp/eval \
  --test-weights /mnt/fcav/self_training/final_results/localization_error_no_fn_with_20fp/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_tp_preds_no_fn_with_20fp/generalized_rcnn
```

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/localization_error_no_fn/eval \
  --test-weights /mnt/fcav/self_training/final_results/localization_error_no_fn/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_tp_preds_no_fn/generalized_rcnn
```

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/localization_error_no_fn_with_10fp/eval \
  --test-weights /mnt/fcav/self_training/final_results/localization_error_no_fn_with_10fp/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_tp_preds_no_fn_with_10fp/generalized_rcnn
```

new-scheme

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/new_training_scheme/eval1024 \
  --test-weights /mnt/fcav/self_training/ws-mani/ws-training-scheme/FRCNN_GTA8k_KITTI-FTL/bs1024
```

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/new_training_scheme/eval2048 \
  --test-weights /mnt/fcav/self_training/ws-mani/ws-training-scheme/FRCNN_GTA8k_KITTI-FTL/bs2048
```

```
container-fn detectron-faster-rcnn-eval \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/new_training_scheme/eval2048_bbox_weights \
  --start-checkpoint 1 \
  --test-weights /mnt/fcav/self_training/ws-mani/ws-training-scheme/FRCNN_GTA8k_KITTI-FTL/bs2048_bbox_weights
```

### detectron-faster-rcnn-feedforward (generate region proposals or prediction)

#### Region proposals

on GTA 8k

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_GTA_train8k.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

on cityscapes

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_cityscapes.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

on KITTI prediction

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_train_prediction.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_train_prediction_ftl.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_object_train_prediction.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

on KITTI val 1000
```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_eval_1k.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

on KITTI train

```
container-fn detectron-faster-rcnn-feedforward  \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_train.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

on instances_caronly_tp_preds

```
container-fn detectron-faster-rcnn-feedforward  \
  --dataset-path /mnt/fcav/self_training/final_results/dataset \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

```
container-fn detectron-faster-rcnn-feedforward  \
  --dataset-path /mnt/fcav/self_training/final_results/dataset \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds_no_fn.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

```
container-fn detectron-faster-rcnn-feedforward  \
  --dataset-path /mnt/fcav/self_training/final_results/dataset \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds_no_fn_with_10fp.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

```
container-fn detectron-faster-rcnn-feedforward  \
  --dataset-path /mnt/fcav/self_training/final_results/dataset \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds_no_fn_with_15fp.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

```
container-fn detectron-faster-rcnn-feedforward  \
  --dataset-path /mnt/fcav/self_training/final_results/dataset \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds_no_fn_with_20fp.yaml \
  --output-path /mnt/fcav/self_training/final_results/region_proposals_GTA200k/proposals_1242 \
  --test-weights /mnt/fcav/self_training/final_results/lowerbound/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

#### Predictions

```
container-fn detectron-faster-rcnn-predictions \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_R-50-FPN_2x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/check_R_50_X_101/R50 \
  --test-weights /mnt/fcav/self_training/final_results/check_R_50_X_101/R50/model_final.pkl
```

```
container-fn detectron-faster-rcnn-predictions \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/check_R_50_X_101/X101 \
  --test-weights /mnt/fcav/self_training/final_results/check_R_50_X_101/X101/model_final.pkl
```

```
container-fn detectron-faster-rcnn-predictions \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/check_R_50_X_101/X101  \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_iter34999.pkl
```

```
container-fn detectron-faster-rcnn-predictions \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000_not_aug_not_scale.yaml \
  --output-path /mnt/fcav/self_training/final_results/lowerbound/prediction35k_not_aug_not_change_scale \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_iter34999.pkl
```

upperbound1

```
container-fn detectron-faster-rcnn-predictions  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound1_eval.yaml \
  --output-path /mnt/fcav/self_training/object_detection/upperbound1_new/predictions_2048 \
  --test-weights /mnt/fcav/self_training/object_detection/upperbound1_new/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_train/generalized_rcnn/model_final.pkl
```

GTA8k+KITTI with prediction+FTL

```
container-fn detectron-faster-rcnn-predictions  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/object_detection/baseline_ss_0.99_FTL/predictions_1920 \
  --test-weights /mnt/fcav/self_training/object_detection/baseline_ss_0.99_FTL/train/voc_GTA_caronly_train_sample8000_coco_KITTI_train_with_prediction/generalized_rcnn/model_final.pkl
```

GTA8k+KITTI with prediction

```
container-fn detectron-faster-rcnn-predictions  \
  --config /mnt/fcav/self_training/final_results/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/final_results/baseline_ss/predictions \
  --test-weights /mnt/fcav/self_training/final_results/baseline_ss/train/voc_GTA_caronly_train_sample8000_coco_KITTI_caronly_train_with_prediction/generalized_rcnn/model_final.pkl
```

lowerbound

```
container-fn detectron-faster-rcnn-predictions  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/prediction_final/prediction_2048 \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```

```
container-fn detectron-faster-rcnn-predictions  \
  --config /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI1000.yaml \
  --output-path /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/prediction_final/eval_1242_new_val \
  --test-weights /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train_voc_GTA_caronly_val/generalized_rcnn/model_final.pkl
```
