import  numpy as np



params = {
    # Model backups
    'load_file': None,
    'save_file': None,
    'save_interval' : 10000,

    # Training parameters
    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
    # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}



class Test:
    def __init__(self):
        self.initial_value = 0

    def testFunction(self):
        if not hasattr(self, 'memory'):
            self.memory = 100
            print(self.memory)

    def test(self):
        lista = [-0.5 , -0.5]
        arra = np.array(lista)
        print(arra.shape)
        arrb = arra[np.newaxis, :]
        print(arrb.shape)

    def testRange(self):
        print(np.arange(10))

    def test_newaxis(self):
        x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
        print(x_data)
        print(x_data.shape)

    def test_list(self):
        memory = [[-0.5,-0.5,2.,0.,-0.25,-0.5 ],
                   [-0.25,-0.5,2.,0.,0.,-0.5],
                   [ 0., -0.5, 2., 0., 0.25, -0.5 ]]
        memory = np.array(memory)
        print(memory)
        print()
        print(memory[:, :2])

    def test_list_float(self):
        lista = [450, 450]
        listb = [450/490, 450/490]
        listc = np.array([450, 450])/490
        print(listb)
        print(listc)

    def test_bigger_equal(self):
        lista = [70, 0]
        if lista[0] >= 70:
            print('ok for move ')

    def test_hstack(self):
        a = [1.20,2.20]
        aa = np.array(a)
        b = True
        c = np.hstack((aa,b))
        print(c)

    def test_deque(self):
        # 相当于一个队列
        pass


    def test_reshape(self):
        lista = np.array([[[1, 2, 1], [3, 4, 1]],
                           [[5, 6, 1], [7, 8, 1]],
                           [[9, 10, 1], [11, 12, 1]],
                           [[13, 14, 1], [15, 16, 1]]

                           ])
        print(lista.shape)
        shape = lista.shape
        dim = shape[0] * shape[1] * shape[2]
        listb = lista.reshape((-1,dim))
        print(listb)
        print(listb.shape)

    def test_save_model(self):
        local_cnt = 10000
        if local_cnt > params['train_start'] and local_cnt % params['save_interval'] == 0:
            print('the condition satisfied.')



if __name__ == '__main__':
    test = Test()
    test.test_save_model()



