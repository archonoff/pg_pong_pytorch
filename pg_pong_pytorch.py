import logging
import gym
import pickle
from itertools import count

import numpy as np

from torch import from_numpy, sigmoid
from torch import nn, optim
from torch.nn import functional as F


logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s %(asctime)s %(message)s',
)

logger = logging.getLogger(__name__)


class PongAgent(nn.Module):
    render = False
    model_filename = 'model_pytorch.pkl'

    batch_size = 10                    # every how many episodes to do a param update?
    save_model_frequency = 100         # через сколько игр сохранять модель (т.е. ее параметры)

    def __init__(self):
        super().__init__()

        input_units = 80 * 80                   # input dimensionality: 80x80 grid
        hidden_units = 200                      # number of hidden layer neurons

        # todo попробовать сверточный слой
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

        self.optimizer = optim.SGD(self.parameters(), lr=1e-4)

        self.clean_buffers()

        self.env = gym.make('Pong-v0')

    def clean_buffers(self):
        """Буферы используются для хранения всех состояний, действий и вознаграждений в текущей итерации обучения"""

        self.action_probs = []
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
        x = sigmoid(self.fc2(x))
        return x

    def forward(self, x):
        return self.policy_forward(x)

    def preprocess(self, image):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        image = image[35:195]  # crop
        image = image[::2, ::2, 0]  # downsample by factor of 2
        image[image == 144] = 0  # erase background (background type 1)
        image[image == 109] = 0  # erase background (background type 2)
        image[image != 0] = 1  # everything else (paddles, ball) just set to 1
        return from_numpy(image.astype(np.float32).ravel())

    def encode_action(self, action):
        """env принимает значения 2 и 3 в качестве управляющих"""
        return 2 if action else 3

    def spread_reward(self):
        """В конце раунда последний reward выставляется для всех фреймов ранее (с нулевым вознаграждением)"""

        """
        def discount_rewards(r):
            # take 1D float array of rewards and compute discounted reward
            discounted_r = np.zeros_like(r)
            running_add = 0
            for t in reversed(range(0, r.size)):
                if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
                running_add = running_add * gamma + r[t]            # gammea = 0.99
                discounted_r[t] = running_add
            return discounted_r
        """

        rewards = []
        last_reward = None
        for reward in reversed(self.rewards):
            if reward != 0:
                last_reward = reward
            rewards.append(last_reward)

        self.rewards = rewards

    def run_round(self, state, round):
        """Один раунд игры (до первого пропущенного шарика)"""

        # Цикл по фреймам игры, т.е. по картинке в каждый момент игрового времени
        reward_sum = 0
        for frame in count():
            self.states.append(state)                       # Сохранение состояния

            action_prob = self.policy_forward(state)
            action = 1 if np.random.uniform() else 0             # sample действия из ответа сети

            image, reward, done, info = self.env.step(self.encode_action(action))
            state = self.preprocess(image)

            self.action_probs.append(action_prob)
            self.rewards.append(reward)                     # Сохранение вознаграждение после действия в состоянии

            reward_sum += reward

            if reward != 0:
                # Шарик улетел, выигрыш получен (1 или -1)
                logger.debug(f'Раунд {round:02d}' + (' WIN' if reward == 1 else ''))
                return state, done

    def run_game(self, state):
        """Одна игра (до 21 очка)"""

        # Цикл по раундам
        for round in count(start=1):
            state, done = self.run_round(state, round)

            if done:
                # Один из игроков набрал 21 очко
                score = sum(self.rewards)
                self.spread_reward()
                return state, score

    def update_parameters(self):
        return      # todo
        self.optimizer.zero_grad()   # zero the gradient buffers     # todo нужно делать только после получения reward

        output = agent(input)                     # Вычисления, чтобы получить какой-то результат
        loss = criterion(output, target)        # Тут главное: вычисление значения функции потерь
        loss.backward()

        self.optimizer.step()

    def training_loop(self):
        initial_image = self.env.reset()
        state = self.preprocess(initial_image)

        for game in count(start=1):
            state, score = agent.run_game(state)

            logger.debug(f'Игра {game} завершилась со счетом {score:.0f}')

            # Оптимизация каждые batch_size сыгранных игр
            if game % self.batch_size == 0:
                logger.debug(f'Оптимизация {game / self.batch_size:.0f}')
                self.update_parameters()

            # Каждые save_model_frequency игр модель сохраняется в файл
            if game % self.save_model_frequency == 0:
                logger.info('Сохранение модели в файл')
                self.save_model()

            break           # todo


# todo предзагрузить в модель параметры из numpy версии

if __name__ == '__main__':
    agent = PongAgent.load_model(resume=False)
    agent.training_loop()
