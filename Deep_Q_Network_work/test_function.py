import  numpy as np

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



if __name__ == '__main__':
    test = Test()
    test.testFunction()
    test.test()
    test.testRange()



