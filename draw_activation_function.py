import numpy as np
import matplotlib.pyplot as plt
from utils import *
from activation_function import *

x=np.arange(-5.0,5.0,0.1)
y1=step_function(x)
y2=sigmoid(x)
y3=relu(x)
y4=tanh(x)

plt.plot(x,y1, linestyle="--", label="step")
plt.plot(x,y2,label="signoid")
plt.plot(x,y3,label = "relu")
plt.plot(x,y4, label="tanh")
plt.ylim(-1.2,1.2) # y축 범위
plt.legend()
plt.show()


# python draw_activation_function.py