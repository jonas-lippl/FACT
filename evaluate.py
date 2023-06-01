import argparse
import os

import numpy as np
import pandas as pd

BASE_PATH = "saved_models"
parser = argparse.ArgumentParser(description='PyTorch FACT evaluation')
parser.add_argument('--name', type=str, default='multi-source-single-target-digit-five', metavar='N',
                    help='short description of the experiment you are running')
parser.add_argument('--dataset', type=str, default='digit-five', metavar='N',
                    choices=['digit-five', 'office_caltech_10', 'office'], help='dataset to use')
parser.add_argument('--num_iter', type=int, default=1, metavar='N', help='number of iteration for training')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='epochs per round of training')
parser.add_argument('--rounds', type=int, default=1000, metavar='N', help='rounds of federated learning')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.005, metavar='N', help='learning rate')
parser.add_argument('--num_targets', type=int, default=1, metavar='N', help='number of target clients')
parser.add_argument('--num_excluded', type=int, default=0, metavar='N', help='number of domains that are excluded '
                                                                             'from training')
args = parser.parse_args()


def write_accuracies_to_file(accuracies, file):
    file.write("Models")
    for key in accuracies.keys():
        file.write(f" & {key}")
    file.write(" & Avg \\\\ \n")
    file.write("FACT")
    avg = []
    for target, accuracy_per_target in accuracies.items():
        mean = round(float(np.mean(accuracy_per_target)), 1)
        avg.append(mean)
        std_dev = round(float(np.std(accuracy_per_target)), 1)
        print(f"Accuracy for {target}: {mean} +/- {std_dev}")
        file.write(f" & {mean} $\\pm$ {std_dev}")
    file.write(f" & {round(float(np.mean(avg)), 1)} ")


def main():
    if not os.path.exists("evaluations"):
        os.makedirs("evaluations")
    file_save_evaluation = open(f"evaluations/{args.name}.txt", "w")
    comments = f"experiment_name={args.name}\n" \
               f"dataset={args.dataset}\n" \
               f"num_targets={args.num_targets}\n" \
               f"num_iter={args.num_iter}\n" \
               f"epochs={args.epochs}\n" \
               f"rounds={args.rounds}\n" \
               f"batch_size={args.batch_size}\n" \
               f"lr={args.lr}\n\n\n"
    file_save_evaluation.write(comments)

    name = f"FACT {args.name}, {args.dataset}, epochs={args.epochs}, rounds={args.rounds}, bs={args.batch_size}, lr={args.lr}"

    accuracies = {}

    # Evaluate csv files
    for subdir in os.listdir(f"{BASE_PATH}/{name}"):
        target = subdir.split("_")[0]
        excluded_str = ''
        iteration = int(subdir.split("_")[-1])
        if args.num_excluded > 0:
            excluded = subdir.split("_")[2]
            excluded_str = f" ({excluded} excluded)"
        if iteration == 0:
            accuracies[f"{target}{excluded_str}"] = []
        for file in os.listdir(f"{BASE_PATH}/{name}/{subdir}"):
            if file.split('.')[-1] == 'csv':
                data = pd.read_csv(f"{BASE_PATH}/{name}/{subdir}/{file}")
                min_discrepancy = min(data['discrepancy loss'].tolist()[int(0.1 * args.epochs):])
                accuracy_at_min_disc = data['accuracy'][data['discrepancy loss'].tolist().index(min_discrepancy)]
                accuracies[f"{target}{excluded_str}"].append(accuracy_at_min_disc * 100)

    # Write accuracies to file
    write_accuracies_to_file(accuracies, file_save_evaluation)

    file_save_evaluation.close()


if __name__ == '__main__':
    main()
