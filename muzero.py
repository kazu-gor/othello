import importlib
import numpy as np
import os

import torch


class MuZero:
    def __init__(self, game_name, config=None, split_resource_in=1):
        """
        Args:
            game_name ([type]): "Othello" etc.
            config ([type], optional): Defaults to None.
            split_resource_in (int, optional): Defaults to 1.

        Raises:
            err: ModuleNotFoundError
        """
        try:
            game_module = importlib.import_module(game_name)
            self.Game = game_module.Game
            self.config = game_module.MuzeroConfig()
        except ModuleNotFoundError as err:
            print(
                f"{game_name} is not a supported game name"
            )
            raise err

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # TODO: Overwrite the config
        # TODO: manage GPU
        # TODO: checkpoint and replay buffer used to initialize workers
        # TODO: Workers

        def train(self, log_in_tensorboard=True):
            """
            Args:
                log_in_tensorboard (bool, optional): [description]. Defaults to True.
            """
            if log_in_tensorboard or self.config.save_model:
                os.makedirs(self.config.results_path, exist_ok=True)


if __name__ == '__main__':
    print("Welcome to muzero for othello!")

    muzero = MuZero("Othello")