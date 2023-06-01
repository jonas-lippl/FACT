: 'experiment_name=multi-source-single-target-digit-five
dataset=digit-five
num_targets=1
num_exclude=0
num_iter=10
epochs=1
rounds=1000
batch_size=128
lr=0.005
lr_decay_rate=0.75
num_identical_domain_clients=1
finetune=yes'


screen -dmS digit-five_evaluation  sh -c 'docker run --gpus \"device=0\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/evaluate.py --dataset=digit-five --num_targets=1 --num_excluded=0 --num_iter=10 --epochs=1 --rounds=1000 --batch_size=128 --lr=0.005 --name=multi-source-single-target-digit-five; exec bash'