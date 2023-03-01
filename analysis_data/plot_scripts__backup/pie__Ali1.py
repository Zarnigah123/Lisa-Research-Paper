"""
Created on Mon Jan 24 14:36:05 2022

@author: nazeela
"""

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

# Data to plot
files__e_var = ['./../BHBH.h5',
                './../NSNS.h5',
                './../NSBH.h5',
                './../BHNS.h5']

files__e_0 = ['./../BHBH.h5',
              './../NSNS.h5',
              './../NSBH.h5',
              './../BHNS.h5']

comp1__e_var = np.ones(4)
comp2__e_var = np.ones(4)
comp3__e_var = np.ones(4)

sizes__e_var, total__e_var = [], []

for i in range(len(files__e_var)):
    with h5.File(files__e_var[i], "r") as f:
        data = f["simulation"][...].squeeze()
    total__e_var.append(len(np.array(data['m1_dco'])))
    size__e_var = total__e_var[i] / 100
    comp__e_var = np.array(data['component'])

    sizes__e_var.append(size__e_var)
    comp1__e_var[i] = (len(np.where(comp__e_var == 0)[0]) / len(comp__e_var)) * size__e_var
    comp2__e_var[i] = (len(np.where(comp__e_var == 1)[0]) / len(comp__e_var)) * size__e_var
    comp3__e_var[i] = (len(np.where(comp__e_var == 2)[0]) / len(comp__e_var)) * size__e_var


labels = ['BHBH', 'NSNS', 'NSBH', 'BHNS']

sizes_comp__e_var = [comp1__e_var[0], comp2__e_var[0], comp3__e_var[0],
                     comp1__e_var[1], comp2__e_var[1], comp3__e_var[1],
                     comp1__e_var[2], comp2__e_var[2], comp3__e_var[2],
                     comp1__e_var[3], comp2__e_var[3], comp3__e_var[3]]

colors = ['#ff6666', '#ffcc99', '#99ff99', 'orange']

colors_comp = ['#c2c2f0', '#ffb3e6', '#ff9999',
               '#c2c2f0', '#ffb3e6', '#ff9999',
               '#c2c2f0', '#ffb3e6', '#ff9999']

# Plot
com_l = [r'Low $\alpha$ disc', r'High $\alpha$ disc','Bulge']
plt.figure(figsize=(8, 8))
plt.pie(sizes__e_var, labels=labels, colors=colors, frame=True, autopct='%1.2f%%', pctdistance=0.85)
patches, texts, _ = plt.pie(sizes_comp__e_var, colors=colors_comp, radius=0.75, pctdistance=0.85, autopct='%1.2f%%')

centre_circle = plt.Circle((0, 0), 0.5, color='black', fc='white', linewidth=0)
plt.annotate(f'Avg. DCOs detected\n= {np.sum(total__e_var)/10000000}%', xy=(-0.45, -0.025))
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.legend(patches, com_l, loc="upper left")
plt.axis('equal')
plt.tight_layout()
# plt.savefig('pie_chart1.pdf')

# #The code for two pie charts
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import math
# fig = plt.figure(figsize=(10, 6))#, constrained_layout=True)
# gs = GridSpec(1, 2, figure=fig)
# axes1 = fig.add_subplot(gs[0, 0])
# axes2 = fig.add_subplot(gs[0, 1])

# sizes__e_0, total__e_0 = [], []

# for i in range(len(files__e_0)):
#     with h5.File(files__e_0[i], "r") as f:
#         data = f["simulation"][...].squeeze()
#     total__e_0.append(len(np.array(data['m1_dco'])))
#     size__e_0 = total__e_0[i] / 100
#     #comp__e_0 = np.array(data['component'])

#     sizes__e_0.append(size__e_0)

# labels = [f'BHBH = {round(sizes__e_0[0]/100000, 8 - int(math.floor(math.log10(abs(sizes__e_0[0]/100000)))) - 1)}%', f'NSNS = {round(sizes__e_0[1]/100000, 8 - int(math.floor(math.log10(abs(sizes__e_0[1]/100000)))) - 1)}%', f'NSBH = {round(sizes__e_0[2]/100000, 8 - int(math.floor(math.log10(abs(sizes__e_0[2]/100000)))) - 1)}%']

# colors = ['#ff9999', '#66b3ff', '#99ff99']
# explode = (0.05, 0.05, 0.05)

# axes2.pie(sizes__e_0, colors=colors, labels=labels, autopct='%1.2f%%', startangle=90, pctdistance=0.65, explode=explode)

# labels = [f'BHBH = {sizes__e_var[0]/100000}%', f'NSNS = {round(sizes__e_var[1]/100000, 8 - int(math.floor(math.log10(abs(sizes__e_var[1]/100000)))) - 1)}%',
#           f'NSBH = {sizes__e_var[2]/100000}%']

# colors = ['#ff9999', '#66b3ff', '#99ff99']



# axes1.pie(sizes__e_var, colors=colors, labels=labels, autopct='%1.2f%%', startangle=90, pctdistance=0.65, explode=explode)

# # Equal aspect ratio ensures that pie is drawn as a circle
# axes1.set_title('Variable eccentricity at ZAMS')
# axes2.set_title('Zero eccentricity at ZAMS')
# axes1.annotate(f'Average total DCOs detected = {np.sum(total__e_var)/10000000}%', xy=(-0.95,1.5))
# axes2.annotate(f'Average total DCOs detected = {np.sum(total__e_0)/10000000}%', xy=(-0.95,1.5))

# axes1.axis('equal')
# axes2.axis('equal')

# # for i in range(3):
# plt.tight_layout()
# # plt.savefig('pie_char__e0_evar.pdf')
