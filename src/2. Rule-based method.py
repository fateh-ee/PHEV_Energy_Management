import matplotlib.pyplot as plt
import numpy as np
import time

# Fitness function parameters
pfu=0.795 # per unit cost of fuel
pel=0.137 # per unit cost of electricity
fpu=1 # per unit fuel consumption
SOC=0.9 # initial value of SoC
soc_values = [SOC] # to store SoC at every seconds during simulation

acc= np.loadtxt('acc_us06.txt') #loading acceleration data of driving cycle

#Applying Rule-based method
const_acc=0.009 #acc constant to consider it's affect on soc charging/discharging and cost function
J=0 # initial cost before starting the car

start_time=time.time()
for a in acc:
    
    # 1. increasing speed
    if a > 0: 
        if SOC > 0.30: # SOC > lower_limit, so battery provides power only
            SOC-=(a*const_acc) # soc discharging rate
            print(f'Engine off 		SOC:{SOC:.2f}		acc:{a:.2f}')
            
        else: # SOC < lower_limit, so engine is turned on which provides power to the vehicle and also charges battery
            SOC+=(a*const_acc)
            print(f'Engine on 		SOC:{SOC:.2f}		acc:{a:.2f}')
            J+=pfu*fpu*a # considering fuel consumption in cost function at each time while the engine is on
            
    # 2. decreasing speed
    else: 
        if SOC > 0.30: # battery > lower_limit, engine is off, so no need to turn on regenarative braking
            SOC+=0
            print(f'Engine idle 		SOC:{SOC:.2f}		acc:{a:.2f}')
            
        else: # battery < lower_limit, so regenarative braking is turned on that can charge the battery
            SOC+=a*const_acc*(-1)# Regenerative braking charges battery. Here, '-1' is multiplied to consider the affect of charging soc (increasing soc value) during negative accele
            print(f'Regenerative Breaking 	SOC:{SOC:.2f}		acc:{a:.2f}')
    soc_values.append(SOC) # storing SOC values at each time step to generate plot
end_time=time.time()

J= J + SOC * pel # considering final value of SOC according to our cost function equation

elapsed_time= end_time-start_time
print(f'Elapsed time:{elapsed_time:.2f}')

print('Final Cost: {:.2f}'.format(J))
print(f'Acceleration size: {len(acc):} seconds ')

plt.plot(list(range(len(soc_values))), soc_values)
plt.title('State of Charge profile')
plt.xlabel('Time(seconds)')
plt.ylabel('SOC')
plt.grid(True)
plt.ylim(0, 0.9)
plt.xlim(0)
plt.show()

#np.savetxt('soc_rl_unece.txt',soc_values) #to save the values of soc for specific driving cycle