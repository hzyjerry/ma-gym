import argparse
import ma_gym
from ma_gym.wrappers import Monitor
from multiprocessing import Pool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='Checkers-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    env = ma_gym.make(args.env)
    env = Monitor(env, directory='recordings/' + args.env, force=True)


    def agent_policy(agent_env):  # this could be any method that interacts with the agent env such as train, test, etc.
        agent_env.reset()
        done = False
        agent_rewards = 0
        while not done:
            obs, reward, done, info = agent_env.step(agent_env.action_space.sample())
            agent_rewards += reward
        return agent_rewards


    for ep_i in range(args.episodes):
        env.seed(ep_i)
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        with Pool(env.n_agents) as p:
            rewards = p.map(agent_policy, env.agents())
            ep_reward += sum(ep_reward)
        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()
