import logging
import gym
import pickle
from itertools import count

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(message)s',
)

logger = logging.getLogger(__name__)


class PongAgent(nn.Module):
    render = False
    model_filename = 'model_pytorch.pkl'

    batch_size = 10                    # every how many episodes to do a param update?
    save_model_frequency = 100         # через сколько игр сохранять модель (т.е. ее параметры)
    learning_rate = 1e-4
    discounting_gamma = 0.99

    last_game = 0

    def __init__(self):
        super().__init__()

        input_units = 80 * 80                   # input dimensionality: 80x80 grid
        hidden_units = 200                      # number of hidden layer neurons

        # todo попробовать сверточный слой
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        self.clean_buffers()

        self.env = gym.make('Pong-v0')

    def clean_buffers(self):
        """Буферы используются для хранения всех состояний, действий и вознаграждений в текущей итерации обучения"""

        self.action_probs = []
        self.actions = []
        self.states = []
        self.rewards = []

    @classmethod
    def load_model(cls, resume=True):
        """Предзагрузка модели с параметрами"""

        try:
            if not resume:
                raise FileNotFoundError
            with open(cls.model_filename, 'rb') as f:
                logger.info(f'Загрузка модели из файла {cls.model_filename}')
                agent = pickle.load(f)
        except FileNotFoundError:
            logger.info('Создание новой модели')
            agent = cls()

        return agent

    def save_model(self):
        with open(self.model_filename, 'wb') as f:
            pickle.dump(self, f)

    def policy_forward(self, x):
        """Одинарный проход по сети"""
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def forward(self, x):
        return self.policy_forward(x)

    @staticmethod
    def preprocess(image, old_state=None):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """

        image = image[35:195]  # crop
        image = image[::2, ::2, 0]  # downsample by factor of 2
        image[image == 144] = 0  # erase background (background type 1)
        image[image == 109] = 0  # erase background (background type 2)
        image[image != 0] = 1  # everything else (paddles, ball) just set to 1

        new_state = torch.from_numpy(image.astype(np.float32).ravel())
        if old_state is None:
            old_state = torch.zeros_like(new_state)

        return new_state - old_state

    @staticmethod
    def encode_action(action):
        """env принимает значения 2 и 3 в качестве управляющих"""
        return 2 if action else 3

    def discount_reward(self):
        for i in range(len(self.rewards) - 1, -1, -1):
            if self.rewards[i] != 0:        # Последний элемент списка не должен быть равен нулю, или будет ошибка
                continue
            self.rewards[i] = self.rewards[i + 1] * self.discounting_gamma

    def run_round(self, state, round):
        """Один раунд игры (до первого пропущенного шарика)"""

        # Цикл по фреймам игры, т.е. по картинке в каждый момент игрового времени
        for frame in count():
            if self.render:
                self.env.render()
            self.states.append(state)                       # Сохранение состояния

            action_prob = self.policy_forward(state)
            action = torch.bernoulli(action_prob)           # sample действия из ответа сети
            self.action_probs.append(action_prob)
            self.actions.append(action)

            old_state = state
            image, reward, done, info = self.env.step(self.encode_action(action))
            state = self.preprocess(image, old_state)

            self.rewards.append(reward)                     # Сохранение вознаграждение после действия в состоянии

            if reward != 0:
                # Шарик улетел, выигрыш получен (1 или -1)
                logger.debug(f'Раунд {round:02d}' + (' WIN' if reward == 1 else ''))
                return state, reward, done

    def run_game(self, state):
        """Одна игра (до 21 очка)"""

        # Цикл по раундам
        reward_sum = 0
        for round in count(start=1):
            state, reward, done = self.run_round(state, round)

            reward_sum += reward

            if done:
                # Один из игроков набрал 21 очко
                return state, reward_sum

    def update_parameters(self):
        self.discount_reward()

        self.optimizer.zero_grad()

        actions = torch.tensor(self.actions).float()
        action_probs = torch.tensor(self.action_probs, requires_grad=True)
        sampled_action_probs = torch.abs(actions - action_probs)        # Вероятности реально соверешнных действий

        loss = (-torch.log(sampled_action_probs) * torch.tensor(self.rewards)).mean()
        loss.backward()

        self.optimizer.step()

        self.clean_buffers()

    def training_loop(self):

        for game in count(start=self.last_game + 1):
            image = self.env.reset()
            state = self.preprocess(image)
            state, score = agent.run_game(state)

            logger.info(f'Игра {game} завершилась со счетом {score:.0f}')
            self.last_game = game

            # Оптимизация каждые batch_size сыгранных игр
            if game % self.batch_size == 0:
                logger.info(f'Оптимизация {game / self.batch_size:.0f}')
                self.update_parameters()

            # Каждые save_model_frequency игр модель сохраняется в файл
            if game % self.save_model_frequency == 0:
                logger.info('Сохранение модели в файл')
                self.save_model()


# todo предзагрузить в модель параметры из numpy версии

if __name__ == '__main__':
    agent = PongAgent.load_model()
    agent.training_loop()
