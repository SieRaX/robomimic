multi_step="att" # one of: single, multi, att

if [ "$multi_step" = "single" ]; then
    subpath="single_step"
    extra_args=""
elif [ "$multi_step" = "multi" ]; then
    subpath="multi_step" 
    extra_args="--multi_step"
elif [ "$multi_step" = "att" ]; then
    subpath="att_step"
    extra_args="--multi_step"
else
    echo "Invalid multi_step value. Must be one of: single, multi, att"
    exit 1
fi

python multistep_bc_rollout.py --agent ../Affect_of_Control_horizon_multistep_BC/horizon_30/20250211080613/models/model_epoch_400_Lift_success_1.0.pth --n_rollouts 100 --video_path ../Affect_of_Control_horizon_multistep_BC/horizon_30/20250211080613/rollout/${subpath}/rollout.mp4 --dataset_path ../Affect_of_Control_horizon_multistep_BC/horizon_30/20250211080613/rollout/${subpath}/rollout.hdf5 --video_skip 5 ${extra_args}