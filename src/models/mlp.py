import torch
import torch.nn as nn
import torch.optim as optim

from models.model_base import ModelStrategy

from omegaconf import DictConfig


class MLP(ModelStrategy):
    def __init__(self):
        super().__init__()

    def build(self, config: DictConfig) -> nn.Module:
        layers = []
        self.config = config
        input_size = 300 * 300 * 3  # Example for resized images
        for hidden_size in config.model.layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 91))  # COCO classes + background

        model = nn.Sequential(*layers)
        return model

    def train(self, model, train_loader, epochs, device=None):
        if device is None:
            device = torch.device("cpu")

        criterion = nn.CrossEntropyLoss()
        match self.config.optimizer.type:
            case "Adam":
                optimizer = optim.Adam(
                    model.parameters(), lr=self.config.optimizer.learning_rate
                )
            case "SGD":
                optimizer = optim.SGD(
                    model.parameters(), lr=self.config.optimizer.learning_rate
                )
            case _:
                raise ValueError(
                    f"Unsupported optimizer type: {self.config.optimizer.type}"
                )

        model.train()
        for epoch in range(epochs):
            batch_count = 0
            for inputs, labels in train_loader:
                # inputs がリスト（画像）の場合、最初の画像を処理
                if isinstance(inputs, list) and len(inputs) > 0:
                    batch_count += 1
                    # リストの画像をテンソルに統合できない場合、ここでスキップ
                    # または、画像を個別に処理する
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Batch {batch_count}: Loaded {len(inputs)} images on {device}"
                    )
                    if batch_count >= 3:  # サンプル: 最初の3バッチだけ処理
                        break
                    continue

                # テンソルをデバイスに移動
                inputs = inputs.to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(
                f"Epoch {epoch + 1}/{epochs} completed - Processed {batch_count} batches"
            )

    def predict(self, model, input_data, device=None):
        if device is None:
            device = torch.device("cpu")

        model.eval()
        with torch.no_grad():
            input_data = input_data.to(device)
            outputs = model(input_data)
            _, predicted = torch.max(outputs, 1)
        return predicted
