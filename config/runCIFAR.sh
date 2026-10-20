ics=(
"0 1 0 0 1 0 1 0 0 0 1 0 0 0"
"0 0 0 1 0 1 0 1 0 1 0 1 0 0"
"0 0 1 0 0 1 1 0 0 1 0 0 0 0"
"0 0 1 0 1 0 1 0 1 0 0 1 0 0"
"0 0 1 0 1 0 1 0 1 0 1 0 1 0"
)
# G-gray   GN-Gaussian Noise    SN-Speckle Noise 
datasets=(
"cifar10G_1"
"cifar10G_2"
"cifar10G_3"
"cifar10G_4"
"cifar10G_5"
)
for idx in "${!ics[@]}"
do
    ic="${ics[$idx]}"
    dataset="${datasets[$idx]}"

    num_ic=$(echo $ic | tr ' ' '\n' | grep -c '^1$')

    attr_ic=""
    for ((i=1;i<num_ic;i++))
    do
        attr_ic="$attr_ic -1"
    done
    attr_ic="$attr_ic 0"
    python train_networks.py \
        --attack attacker1 \
        --dataset $dataset \
        --model vgg \
        --add_ic_attack $ic \
        --add_ic_victim $ic \
        --gpuid 2 \
        --seed 1022 \
        --cov_lambda 1.5 \
        --target_attr  0 \
        --smooth_upper 0.6 \
        --attr_ic $attr_ic  \
        --attr_victim $attr_ic  \
        --attr_num 2 \
        --need_smooth True \
        --need_cov True \
        --thres 0.95 0.9 0.8 \
        --save_path ./new_results2/cifar.json
    python asr_test.py \
        --attack attacker1 \
        --dataset $dataset \
        --model vgg \
        --add_ic_attack $ic \
        --add_ic_victim $ic \
        --gpuid 3 \
        --seed 1022 \
        --cov_lambda 1.5 \
        --target_attr  0 \
        --smooth_upper 0.6 \
        --attr_ic $attr_ic  \
        --attr_victim $attr_ic  \
        --attr_num 2 \
        --need_smooth True \
        --need_cov True \
        --thres 0.95 0.9 0.8 \
        --save_path ./new_results2/cifar.json
done