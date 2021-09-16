export CUDA_VISIBLE_DEVICES=0
export LAYERS=64
export LR=0.0001
export LOC="wells-1_microns-30_interp-40-64-64"
# export LOC="wells-1_microns-30_interp-14-64-64_threshold_subims-32"
export LOGDIR="/home/connor/GDrive/SCGSR/data/prnn-logs/"
export DATADIR="/home/connor/GDrive/SCGSR/data/datasets/"
export CHECKDIR="/home/connor/GDrive/SCGSR/data/prnn-checkpoints/"
export RESDIR="/home/connor/GDrive/SCGSR/data/prnn-results/"
mkdir -p $LOGDIR$LOC

echo Data from $DATADIR$LOC
echo Logs at $LOGDIR$LOC
echo Results at $DATRESDIR$LOC"_layers-"$LAYERS"_lr-"$LR
echo Model checkpoints at $CHECKDIR$LOC"_layers-"$LAYERS"_lr-"$LR
cd ..
(python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths $DATADIR$LOC-train.npz \
    --valid_data_paths $DATADIR$LOC-valid.npz \
    --save_dir $CHECKDIR$LOC"_layers-"$LAYERS"_lr-"$LR \
    --gen_frm_dir $RESDIR$LOC"_layers-"$LAYERS"_lr-"$LR \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 3 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden $LAYERS,$LAYERS,$LAYERS,$LAYERS \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr $LR \
    --batch_size 8 \
    --max_iterations 6000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 1000 \
) 2>&1 | tee $LOGDIR$LOC"_layers-"$LAYERS"_lr-"$LR/pnn.log
