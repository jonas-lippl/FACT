import os
import random
import torch
import pandas as pd


class Server:
    """Server class to simulate the federated learning process using the FACT algorithm.

    Attributes:
        name: String that describes the experiment.
        iteration: Number of iteration of experiment if it is run for multiple times.
        target: Name of target domain.
        source_clients: List of at least two source clients with labeled data.
        target_clients: List of target clients with unlabeled data.
        feature_model: Model that works as a feature extractor.
        predictor_model: Model that takes output from feature_model and predicts a label.
        rounds_of_fed_learning: Number of rounds the federated learning process is repeated.
        epochs: Number of epochs the models are trained at each client.
        finetune: Boolean to indicate if the classifiers are finetuned after the source training step.
    """

    def __init__(
            self,
            name,
            iteration,
            target,
            source_clients,
            target_clients,
            feature_model,
            predictor_model,
            rounds_of_fed_learning,
            epochs,
            finetune,
    ):
        self.name = name
        self.iteration = iteration
        self.target = target
        self.source_clients = source_clients
        self.target_clients = target_clients
        self.rounds_of_fed_learning = rounds_of_fed_learning
        self.epochs = epochs
        self.finetune = finetune
        self.G = feature_model
        self.C2 = predictor_model
        self.number_of_target_clients = len(target_clients)
        self.C1_client_history = []
        self.C2_client_history = []

    def run(self):
        for learning_round in range(self.rounds_of_fed_learning):
            print(f"\nRound: {learning_round + 1}")

            source_updates_G = []
            source_updates_C = []
            target_updates_G = []
            target_updates_C = []

            selected_source_clients = random.sample(self.source_clients, 2)
            for source_client in selected_source_clients:
                G_update, C_update = source_client.train_at_client(
                    self.G.state_dict(), self.C2.state_dict()
                )
                source_updates_C.append(C_update)
                source_updates_G.append(G_update)
            aggregated_source_update_G = self.aggregate_updates(self.G, source_updates_G, [1, 1])

            if self.finetune:
                source_updates_C = []
                for source_client in selected_source_clients:
                    C_update = source_client.finetune_classifiers(aggregated_source_update_G, epochs=1)
                    source_updates_C.append(C_update)
            for _ in range(self.epochs):
                self.C1_client_history.append(selected_source_clients[0].name)
                self.C2_client_history.append(selected_source_clients[1].name)

            # Target clients
            for client in self.target_clients:
                G_update, C_update_1, C_update_2 = client.train_at_client(
                    aggregated_source_update_G, *source_updates_C
                )
                target_updates_G.append(G_update)
                target_updates_C.append(C_update_1)
                target_updates_C.append(C_update_2)

            aggregated_update_G = self.aggregate_updates(self.G, updates=target_updates_G,
                                                         weights=[1 / self.number_of_target_clients] * self.number_of_target_clients)

            aggregated_update_C2 = self.aggregate_updates(self.C2, target_updates_C, [1 for _ in range(len(target_updates_C))])

            self.G.load_state_dict(aggregated_update_G)
            self.C2.load_state_dict(aggregated_update_C2)

        self.save_model()

    @staticmethod
    def _sum_up_dict_values(model_params, key, updates, weights):
        summed_dict_values = torch.zeros_like(model_params[key], dtype=torch.float)
        for update, w in zip(updates, weights):
            summed_dict_values += w * update[key]
        return summed_dict_values

    def aggregate_updates(self, model, updates, weights):
        model_params = model.state_dict()
        for key in model_params.keys():
            model_params[key] = self._sum_up_dict_values(
                model_params, key, updates, weights
            ) / (sum(weights))
        return model_params

    def save_model(self):
        self.save_target_client_data()
        torch.save(
            self.G.state_dict(), f"saved_models/{self.name}/{self.target}_{self.iteration}/G.pth"
        )
        torch.save(
            self.C2.state_dict(), f"saved_models/{self.name}/{self.target}_{self.iteration}/C.pth"
        )

    def save_target_client_data(self):
        if not os.path.exists(f"saved_models/{self.name}/{self.target}_{self.iteration}"):
            os.makedirs(f"saved_models/{self.name}/{self.target}_{self.iteration}")
        for target_client in self.target_clients:
            if target_client.loss_dis and target_client.accuracy and target_client.loss:
                df = pd.DataFrame(zip(range(len(target_client.accuracy)), target_client.loss_dis,
                                      target_client.accuracy, target_client.loss,
                                      self.C1_client_history, self.C2_client_history),
                                  columns=['epoch', 'discrepancy loss', 'accuracy',
                                           'test loss', 'head 1', 'head 2'])
                df.to_csv(
                    f"saved_models/{self.name}/{self.target}_{self.iteration}"
                    f"/loss_dis_vs_accuracy_{target_client.name}.csv",
                    index=False,
                )
