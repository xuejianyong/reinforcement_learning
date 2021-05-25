from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):  # 一共进行300次的episode
        print('---------------------------- episode: %d ----------------------------' % episode)
        observation = env.reset()  # [-0.5 -0.5] 这样做是为了在神经网络当中的值控制在[0,1]之间  # initial observation 获得初始agent在环境当中的初始状态
        while True:
            env.render()  # fresh env
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)  # 保存每一步的交互信息：s,a,r,s_
            if (step > 200) and (step % 5 == 0):
                RL.learn()  # 在进行到200步的时候，这一步比较重要
            observation = observation_
            if done:
                break
            step += 1
    print('current step is: %d' % step)
    print('over')  # end of game
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(
        env.n_actions,
        env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
        # output_graph=True
        )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()