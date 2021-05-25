import numpy as np

memory = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])
print(memory.shape)
sample_index = np.random.choice(memory.shape[0], size=2)
print(sample_index)
batch_memory = memory[sample_index]
print(batch_memory)
s = batch_memory[:, -2:]
print(s)
