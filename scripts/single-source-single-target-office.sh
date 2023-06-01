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


screen -dmS office_amazon_0_1_1000 sh -c 'docker run --gpus \"device=0\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/FACT.py --dataset=office --target=amazon --exclude_domains=dslr --num_iter=0 --epochs=1 --rounds=1000 --batch_size=128 --lr=0.005 --name=single-source-single-target-office --test_source=no --test_target=yes --lr_decay_rate=0.75 --num_identical_domain_clients=2 --finetune=yes; exec bash'
screen -dmS office_amazon_0_1_1000 sh -c 'docker run --gpus \"device=1\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/FACT.py --dataset=office --target=amazon --exclude_domains=webcam --num_iter=0 --epochs=1 --rounds=1000 --batch_size=128 --lr=0.005 --name=single-source-single-target-office --test_source=no --test_target=yes --lr_decay_rate=0.75 --num_identical_domain_clients=2 --finetune=yes; exec bash'
screen -dmS office_dslr_0_1_1000 sh -c 'docker run --gpus \"device=2\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/FACT.py --dataset=office --target=dslr --exclude_domains=amazon --num_iter=0 --epochs=1 --rounds=1000 --batch_size=128 --lr=0.005 --name=single-source-single-target-office --test_source=no --test_target=yes --lr_decay_rate=0.75 --num_identical_domain_clients=2 --finetune=yes; exec bash'
screen -dmS office_dslr_0_1_1000 sh -c 'docker run --gpus \"device=3\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/FACT.py --dataset=office --target=dslr --exclude_domains=webcam --num_iter=0 --epochs=1 --rounds=1000 --batch_size=128 --lr=0.005 --name=single-source-single-target-office --test_source=no --test_target=yes --lr_decay_rate=0.75 --num_identical_domain_clients=2 --finetune=yes; exec bash'
screen -dmS office_webcam_0_1_1000 sh -c 'docker run --gpus \"device=4\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/FACT.py --dataset=office --target=webcam --exclude_domains=amazon --num_iter=0 --epochs=1 --rounds=1000 --batch_size=128 --lr=0.005 --name=single-source-single-target-office --test_source=no --test_target=yes --lr_decay_rate=0.75 --num_identical_domain_clients=2 --finetune=yes; exec bash'
screen -dmS office_webcam_0_1_1000 sh -c 'docker run --gpus \"device=5\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/FACT.py --dataset=office --target=webcam --exclude_domains=dslr --num_iter=0 --epochs=1 --rounds=1000 --batch_size=128 --lr=0.005 --name=single-source-single-target-office --test_source=no --test_target=yes --lr_decay_rate=0.75 --num_identical_domain_clients=2 --finetune=yes; exec bash'
