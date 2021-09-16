export CUDA_VISIBLE_DEVICES=0
export LOC="DeCamp_video-1_interp-280-128-128_threshold"
export LOGDIR="/home/connor/GDrive/SCGSR/data/prnn-logs/"
export DATADIR="/home/connor/GDrive/SCGSR/data/datasets/"
export CHECKDIR="/home/connor/GDrive/SCGSR/data/prnn-checkpoints/"
export RESDIR="/home/connor/GDrive/SCGSR/data/prnn-results/"
mkdir -p $LOGDIR$LOC

echo Data from $DATADIR$LOC
echo Logs at $LOGDIR$LOC
echo Results at $DATADIR$LOC
echo Model checkpoints at $CHECKDIR$LOC
cd ..
(python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths $DATADIR$LOC-train.npz \
    --valid_data_paths $DATADIR$LOC-valid.npz \
    --save_dir $CHECKDIR$LOC \
    --gen_frm_dir $RESDIR$LOC \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 128 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.0003 \
    --batch_size 8 \
    --max_iterations 800 \
    --display_interval 100 \
    --test_interval 200 \
    --snapshot_interval 400 \
) 2>&1 | tee $LOGDIR$LOC/pnn.log
