export CUDA_VISIBLE_DEVICES=0

# Neural net parameters
export LAYERS="64"
export LR="0.0003"

# Save directories
export LOGDIR="results/logs/"
export DATADIR="data/datasets/"
export CHECKDIR="results/checkpoints/"
export RESDIR="results/validation_images/"

# Specific case
export LOC="nufeb_interp-20-100-100_threshold"

# Make log directory
if [!-d $LOGDIR]; then
  mkdir $LOGDIR;
fi;

echo Data from $DATADIR$LOC
echo Logs at $LOGDIR$LOC.log
echo Results at $DATRESDIR$LOC"_layers-"$LAYERS"_lr-"$LR
echo Model checkpoints at $CHECKDIR$LOC"_layers-"$LAYERS"_lr-"$LR
(python -u predrnn-pytorch/run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths $DATADIR$LOC-train.npz \
    --valid_data_paths $DATADIR$LOC-valid.npz \
    --save_dir $CHECKDIR$LOC"_layers-"$LAYERS"_lr-"$LR \
    --gen_frm_dir $RESDIR$LOC"_layers-"$LAYERS"_lr-"$LR \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 100 \
    --img_channel 1 \
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
    --max_iterations 20000 \
    --display_interval 500 \
    --test_interval 1000 \
    --snapshot_interval 1000 \
) 2>&1 | tee $LOGDIR$LOC"_layers-"$LAYERS"_lr-"$LR.log
