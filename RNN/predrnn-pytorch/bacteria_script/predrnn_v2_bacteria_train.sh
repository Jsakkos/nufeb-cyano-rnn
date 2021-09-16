export CUDA_VISIBLE_DEVICES=0
# export LOC="wells-1_microns-30_interp-14-64-64"
export LOC="wells-1_microns-30_interp-14-64-64_threshold"
# export LOC="wells-1_microns-30"
export LOGDIR="/home/connor/GDrive/SCGSR/data/prnn2-logs/"
export DATADIR="/home/connor/GDrive/SCGSR/data/datasets/"
export CHECKDIR="/home/connor/GDrive/SCGSR/data/prnn2-checkpoints/"
export RESDIR="/home/connor/GDrive/SCGSR/data/prnn2-results/"
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
    --model_name predrnn_memory_decoupling \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 3 \
    --input_length 7 \
    --total_length 14 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 250 \
    --r_sampling_step_2 500 \
    --r_exp_alpha 2500 \
    --batch_size 8 \
    --lr 0.0003 \
    --max_iterations 1000 \
    --display_interval 100 \
    --test_interval 200 \
    --snapshot_interval 500 \
) 2>&1 | tee $LOGDIR$LOC/pnn2.log
