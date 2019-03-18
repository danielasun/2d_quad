import matplotlib.pyplot as plt
import numpy as np

# sigmoid
k1 = 1
k2 = 1
k3 = 1

# arc tan
a1 = 1
a2 = 1
a3 = 1


plt.figure()
s = np.linspace(-1,1, 100)
plt.plot(s, k1*(np.exp(-k2*s + k3)-1)/(np.exp(-k2*s + k3)+1))
plt.xlabel('s')
plt.grid()
plt.title('sigmoid')
plt.show()

plt.figure()
s = np.linspace(-np.pi/2,np.pi/2, 100)
plt.plot(s,-a1*np.arctan(a2*s + a3))
plt.xlabel('s')
plt.grid()
plt.title('atan')
plt.show()

plt.figure()
s = np.linspace(-10,10, 100)
plt.plot(s,1 + 1/(1+np.exp(-s)))
plt.xlabel('s')
plt.grid()
plt.title('H safety function')
plt.show()
