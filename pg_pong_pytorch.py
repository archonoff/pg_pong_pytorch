# Pytorch версия

import gym

from torch import nn, optim
from torch.nn import functional as F


# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid


class Net(nn.Module):

    def __init__(self, env):
        super().__init__()


        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, 1)

        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

        self.env = env


    def save_parameters(self):
        raise NotImplementedError

    def load_parameters(self):
        raise NotImplementedError

    def policy_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

    def prepro(self, image):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        image = image[35:195]  # crop
        image = image[::2, ::2, 0]  # downsample by factor of 2
        image[image == 144] = 0  # erase background (background type 1)
        image[image == 109] = 0  # erase background (background type 2)
        image[image != 0] = 1  # everything else (paddles, ball) just set to 1
        return image.astype(np.float).ravel()

    def forward(self, x):
        # todo Тут должен быть полный проход?
        # Вызывается при вызове call на экземпляре класса
        # x = self.policy_forward(x)

        observation = self.env.reset()

        return x


env = gym.make("Pong-v0")

# todo предзагрузка параметров из файла

net = Net(env)
print(net)
# print(list(net.parameters()))

exit()

n = 0
while n < 10:
    n += 1

    net.optimizer.zero_grad()   # zero the gradient buffers     # todo нужно делать только после получения reward

    output = net(input)                     # Вычисления, чтобы получить какой-то результат
    loss = criterion(output, target)        # Тут главное: вычисление значения функции потерь
    loss.backward()

    net.optimizer.step()    # Does the update


exit()


""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle

if resume:
    with open('save.p', 'rb') as f:
        model = pickle.load(f)
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'resetting env. episode reward total was {reward_sum}. running mean: {running_reward}')

        # Каждые 100 эпизодов сохранение параметров на диск
        if episode_number % 100 == 0:
            with open('save.p', 'wb') as f:
                pickle.dump(model, f)

        # Обнуление значений, рестарт окружения
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print(f'ep {episode_number}: game finished, reward: {reward}' + '' if reward == -1 else ' !!!!!!!!')
