python train_net.py \
--config-file configs/vidor_faster_rcnn_R_101_C4_GPU_4.yaml \
--num-gpus 4 \
--resume \
OUTPUT_DIR output/vidor_R_101_C4_$(date +%Y%m%d)_$RANDOM