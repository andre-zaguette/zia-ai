import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any

class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.preferences = self.load_preferences()

    def load_preferences(self) -> Dict[str, Any]:
        file_path = f"data/{self.user_id}_preferences.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return json.load(file)
        return {"personality": {}, "appearance": {}}

    def save_preferences(self) -> None:
        file_path = f"data/{self.user_id}_preferences.json"
        with open(file_path, "w") as file:
            json.dump(self.preferences, file)

    def update_preferences(self, category: str, key: str, value: Any) -> None:
        if category in self.preferences:
            self.preferences[category][key] = value
        else:
            self.preferences[category] = {key: value}
        self.save_preferences()

class PersonalizationModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.interactions = []

    def train(self, data: np.ndarray, labels: np.ndarray):
        self.model.fit(data, labels)

    def predict(self, interaction: np.ndarray) -> int:
        return self.model.predict([interaction])[0]

    def update_model(self, interaction: np.ndarray, result: int):
        self.interactions.append((interaction, result))
        data = np.array([i[0] for i in self.interactions])
        labels = np.array([i[1] for i in self.interactions])
        self.train(data, labels)
