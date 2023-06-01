import os.path
from itertools import combinations

from datasets.utils import get_source_domains_for_dataset

# Define the parameters:
experiment_name = 'single-source-single-target-office'  # Short description of the experiment. Relevant for file names.
dataset = 'office'  # Dataset you want to use. Options: digit-five, office_caltech_10, office
num_targets = 1  # Number of target domains.
num_excluded = 0  # Number of domains you want to exclude from training. Make sure that there are at least two source domains remaining.
num_iter = 1  # Number of iterations the experiment should be repeated. Mean and standard deviation of the accuracies will be calculated.
epochs = 1  # Epochs per round of training.
rounds = 1000  # Rounds of federated learning.
batch_size = 128
lr = 0.005
test_target = 'yes'  # Should the model be tested in each round at the target clients? Options: yes/no
test_source = 'no'  # Should the model be tested at each round at source clients? Options: yes/no
lr_decay_rate = 0.75  # Exponent of the learning rate scheduler according to lr = lr_0 * (1 + 10 * epoch/total_epochs)^(-lr_decay_rate)
num_identical_domain_clients = 2  # Number of clients from the same domain. The datasets are split into equally sized subsets. Each subset is assigned to a client.
finetune = 'yes'  # Should the source Classifier be fine-tuned? Options: yes/no

num_gpus = 8  # set this to the number of gpu's you have
if __name__ == '__main__':
    script = "FACT.py"
    if not os.path.exists("scripts"):
        os.makedirs("scripts")
    file_run_experiment = open(f"scripts/{experiment_name}.sh", "w")
    file_evaluate = open(f"scripts/{experiment_name}_evaluate.sh", "w")
    comments = f": 'experiment_name={experiment_name}\n" \
               f"dataset={dataset}\n" \
               f"num_targets={num_targets}\n" \
               f"num_exclude={num_excluded}\n" \
               f"num_iter={num_iter}\n" \
               f"epochs={epochs}\n" \
               f"rounds={rounds}\n" \
               f"batch_size={batch_size}\n" \
               f"lr={lr}\n" \
               f"lr_decay_rate={lr_decay_rate}\n" \
               f"num_identical_domain_clients={num_identical_domain_clients}\n" \
               f"finetune={finetune}'\n\n\n"

    file_run_experiment.write(comments)
    file_evaluate.write(comments)

    gpu = 0
    for i in range(num_iter):
        domains = get_source_domains_for_dataset(dataset)
        target_sets = combinations(domains, num_targets)
        for target_domains in target_sets:
            td = ",".join(target_domains)
            domains = get_source_domains_for_dataset(dataset)
            for target_domain in target_domains:
                domains.remove(target_domain)
            exclude_sets = combinations(domains, num_excluded)
            for exclude_domains in exclude_sets:
                ed = ",".join(exclude_domains)
                command = f"screen -dmS {dataset}_{td}_{i}_{epochs}_{rounds} sh -c 'docker run --gpus \\\"device={gpu}\\\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/{script} --dataset={dataset} --target={td} --exclude_domains={ed} --num_iter={i} --epochs={epochs} --rounds={rounds} --batch_size={batch_size} --lr={lr} --name={experiment_name} --test_source={test_source} --test_target={test_target} --lr_decay_rate={lr_decay_rate} --num_identical_domain_clients={num_identical_domain_clients} --finetune={finetune}; exec bash\'"
                file_run_experiment.write(command)
                file_run_experiment.write('\n')
                print(command, "\n")
                gpu += 1
                if gpu == num_gpus:
                    gpu = 0
                    file_run_experiment.write('\n')

    file_evaluate.write(
        f"screen -dmS {dataset}_evaluation  sh -c 'docker run --gpus \\\"device={0}\\\"  -it --rm -u `id -u $USER` -v $SOURCE_PATH_FACT:/mnt fact python3 /mnt/evaluate.py --dataset={dataset} --num_targets={num_targets} --num_excluded={num_excluded} --num_iter={num_iter} --epochs={epochs} --rounds={rounds} --batch_size={batch_size} --lr={lr} --name={experiment_name}; exec bash\'")
    file_evaluate.close()
    file_run_experiment.close()
