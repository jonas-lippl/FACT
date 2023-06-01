: 'experiment_name=single-source-single-target-office
dataset=office
num_targets=1
num_exclude=1
num_iter=1
epochs=1
rounds=1000
batch_size=128
lr=0.01
lr_decay_rate=0.75
num_identical_domain_clients=2
finetune=yes'


screen -dmS office_evaluation  sh -c 'docker run --gpus \"device=0\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/evaluate.py --dataset=office --num_targets=1 --num_excluded=1 --num_iter=1 --epochs=1 --rounds=1000 --batch_size=128 --lr=0.005 --name=single-source-single-target-office; exec bash'