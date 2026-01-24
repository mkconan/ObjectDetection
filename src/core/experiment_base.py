from models.model_base import ModelStrategy

from omegaconf import DictConfig
import torch


class ExperimentBase:
    def __init__(self, strategy: ModelStrategy, device=None):
        self.strategy = strategy
        self.device = device if device is not None else torch.device("cpu")

    def run_experiment(self, config: DictConfig, train_loader, epochs, input_data):
        # Build the model using the strategy
        model = self.strategy.build(config)
        model = model.to(self.device)

        # Train the model
        self.strategy.train(model, train_loader, epochs, self.device)

        # Make predictions only if input_data is provided
        if input_data is not None:
            predictions = self.strategy.predict(model, input_data)
            return predictions
        else:
            print("No input data for prediction, skipping inference step")
