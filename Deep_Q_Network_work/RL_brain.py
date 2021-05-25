"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  # 至今还不太明白设置这个参数的意义 testici
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0  # 这个参数用来做什么的呢 testici 用于将eval_net 当中的参数值每一个固定批次replace_target_iter的 更新在target_net当中

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # 一个专门用来存放 experience replay set 经验回放集合，用来存储之前记忆的容器

        # consist of [target_net, evaluate_net]
        # target表示真实得到的内容 这个需要根据具体的实现来看 testici
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        #  assign函数的作用是将 e的值付给t的值，按照道理来说，应该是

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    # 采用tensorflow来定义神经网络的输入，输出，及其网络结构和数据处理流程。
    def _build_net(self):
        # ------------------ evaluate_net ------------------ #
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 其中一个输入，用来得到当前状态下各动作的q-value
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 各个动作的q-value
        with tf.variable_scope('eval_net'):  # 定义变量空间，这样可以结合tf.get_variable来实现变量共享。
            # c_names(collections_names) are the collections to store variables # 开始定义一些layer会用到的参数 # config of layers
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0., 0.3)  # 初始化权重
            b_initializer = tf.constant_initializer(0.1)           # 初始化偏置

            # 开始定义网络结构 网络的第一层 first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):  # 定义第一层网络的变量空间，主要两类参数，一个是权重，另一个是偏置
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            with tf.variable_scope('l2'):  # 定义第二层网络的变量空间，主要两类参数，一个是权重，另一个是偏置
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2  # eval_net 计算出在在 self.s 状态下不同动作的q-value

        with tf.variable_scope('loss'):# 定义了损失的计算过程，也是tensorflow当中的一个变量，只是给出了计算过程
            # squared_difference 指的是(x - y)(x - y)的值，reduce_mean指的是求所有元素的均值
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):  #
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  # 一个优化器模型用来训练模型

        # ------------------ target_net ------------------ #
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 定义了tensorflow当中target网络的输入节点，但是注意在target网络当中的输入为下一个状态。
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # 开始定义网络的结构，网络的第一层 first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):  # 定义第一层网络的变量空间，主要两类参数，一个是权重，另一个是偏置
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            with tf.variable_scope('l2'):  # 定义第二层网络的变量空间，主要两类参数，一个是权重，另一个是偏置
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]  # 由于在网路的设计当中，出入是一个二维的向量，所以要给其加维
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})  # forward feed the observation and get q value for every actions
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:  # 第一个两百次target_iter=300 每300次的迭代来更新一次replace_target_op
            self.sess.run(self.replace_target_op)  # 将eval_net的值映射到target_网络当中，这时候的eval_net当中的参数有哪些内容呢？ testici
            print('\n target_net_params are updated by the eval_net_params \n')

        # 从memory当中取得部分训练样本 # sample batch memory from all memory
        if self.memory_counter > self.memory_size:  # memory_size 这里是2000 由于每一步都会往memory当中存放之前的交互经验，
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)  # batch_size=32 并不是所有的memory都会用来训练，而是一个batch size的memory
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]  # memory当中存储的数据内容为 ((s, [a, r], s_)) 一共有32条数据  32X6

        q_next, q_eval = self.sess.run(  #   # 在以往的记忆当中提取一定的记忆交互样本，来计算网络值  [32,4]
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params 取到sample当中
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()  # [32,4]

        batch_index = np.arange(self.batch_size, dtype=np.int32)  # [0,1,2,..,31]
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # memory当中的动作   (32,)
        reward = batch_memory[:, self.n_features + 1]  # 获取memory当中 s[r]的内容  (32,)

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)  # 根据记忆中提取的样本数据在网络中计算的结果，这样有了 q-target

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



