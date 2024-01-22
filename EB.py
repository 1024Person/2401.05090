"""
    用数值法求解微分方程
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
# ================================== 初始化参数
# 论文中给的数据
omega = 1
omegaL = omega  # 共振情况
Gamma = 0.04*omega  # 电池和充电器都跟一个共同的“池”的耗散相互作用，这个指的既是a的也是b的Γ
J = Gamma/2 # 电池和充电器的耦合强度
Epsilon = 0.1*omega # 外场驱动的振幅
kappa=0.003*omega   # 电池和充电器的耗散
# 根据论文中的关系自己算的+
mu = 1j  # 这个很多东西都还没有搞懂
Lambda = kappa+Gamma # 这个指的既是a的也是b的λ，
# 时间序列
tn = 5000
tlist = np.linspace(0,500,tn)
dt = tlist[1]-tlist[0]

# a，b，ada，bdb，adb的期望值
a = np.zeros_like(tlist)
b = np.zeros_like(tlist)
ada = np.zeros_like(tlist)
bdb = np.zeros_like(tlist)
adb = np.zeros_like(tlist)

for i in range(tn-1):
    # 难道微分方程的数值解法不能用这个有限差分法吗？

    a[i+1] = a[i]+(-(Lambda /2 + 1j*omega)*a[i]-1j*np.exp(-1j*omegaL*tlist[i])*Epsilon)*dt
    b[i+1] = b[i]+(-(Lambda /2 + 1j*omega)*b[i]+np.conj(mu)*Gamma*a[i])*dt
    ada[i+1] = ada[i]+(-Lambda*ada[i]-2*np.imag(np.exp(1j*omegaL*tlist[i])*Epsilon*a[i]))*dt
    bdb[i+1] = bdb[i]+(-Lambda*bdb[i]-2*np.real(mu*Gamma*adb[i]))*dt
    adb[i+1] = adb[i]+(-Lambda*adb[i]+np.conj(mu)*Gamma*ada[i]+1j*np.exp(1j*omegaL*tlist[i])*Epsilon*b[i])*dt
    if np.mod(i,1000)==0:
        print(i)  

plt.figure()
plt.plot(J*tlist,ada,'r',label=r"$E_A^{nr}$")
plt.plot(J*tlist,bdb,'b',label=r"$E_B^{nr}$")
plt.legend()
plt.xlabel('Jt')
plt.xticks([0,2,4,6,8,10],["0","2","4","6","8","10"])
# plt.yticks([0,20,40,60,80],["0","20","40","60","80"])
plt.show()
