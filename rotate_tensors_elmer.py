import pandas as pd
import numpy as np
import sys
sys.path.append('/home/juanpelvis/Documents/python_code/')
sys.path.append('/home/juanpelvis/Documents/python_code/borehole_paper')
#import aux_inclino2 as AUX
import fig_aux as FX
import matplotlib.pyplot as plt
#
def rotate_tensor(tensor,direction):
    """
    Rotates a tensor in the horizontal direction
    In this case we want X following 55°, ergo 90 + 55 = 145
    """
    c, s = np.cos(direction), np.sin(direction)
    if len(tensor.columns) == 6:
        # xx yy zz xy yz yz
        R = np.array(((c, -s, 0),
                      (s,c, 0),
                      (0, 0, 1)))
        Rm= np.linalg.inv(R)
        for i in range(len(tensor)):
            mat = np.array(((tensor.iloc[i][tensor.columns[0]], tensor.iloc[i][tensor.columns[3]], tensor.iloc[i][tensor.columns[5]]),
                            (tensor.iloc[i][tensor.columns[3]],tensor.iloc[i][tensor.columns[1]],tensor.iloc[i][tensor.columns[4]]),
                            (tensor.iloc[i][tensor.columns[5]],tensor.iloc[i][tensor.columns[4]],tensor.iloc[i][tensor.columns[2]])))
            mat = np.dot(Rm,np.dot(mat,R))
            tensor.iloc[i] = [mat[0][0],mat[1][1],mat[2][2],mat[0][1],mat[1][2],mat[0][2],]
    #
    if len(tensor.columns) == 9:
        # xx xy xz yx yy yz zx zy zz
        R = np.array(((c, -s, 0),
                      (s,c, 0),
                      (0, 0, 1)))
        Rm= np.linalg.inv(R)
        for i in range(len(tensor)):
            mat = np.array(((tensor.iloc[i][tensor.columns[0]], tensor.iloc[i][tensor.columns[1]], tensor.iloc[i][tensor.columns[2]]),
                            (tensor.iloc[i][tensor.columns[3]],tensor.iloc[i][tensor.columns[4]],tensor.iloc[i][tensor.columns[5]]),
                            (tensor.iloc[i][tensor.columns[6]],tensor.iloc[i][tensor.columns[7]],tensor.iloc[i][tensor.columns[8]])))
            mat = np.dot(Rm,np.dot(mat,R))
            tensor.iloc[i] = [mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2], mat[2][0], mat[2][1], mat[2][2]]
    tensor.reset_index(inplace=True)
    return tensor

path = '/home/juanpelvis/Documents/Inclino/simulation_outputs/'
gradu = 'GradVitForage2_viscoinv.dat'
graduconst = 'GradVitForage2_viscoconst.dat'
strain_uniform = 'StrainRateForage2.dat'
strain_inv = 'StrainRateForage2_viscoinv.dat'
stress = 'StressForage2.csv'
""" """
list_of_gradvit = [
    'GradVitForage2_Aconst_n=3.dat',
    'GradVitForage2_Aconst_n=4.dat',
    'GradVitForage2_Aconst_n=5.dat',
    'GradVitForage2_Ainv_n=3.dat',]
gradvitcols = ['depth','dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz']
straincols = ['depth','exx','eyy','ezz','exy','eyz','exz']
stresscols = ['depth','sxx','syy','szz','sxy','syz','sxz']
list_of_strain = [
    'StrainRateForage2_Aconst_n=3.dat',
    'StrainRateForage2_Aconst_n=4.dat',
    'StrainRateForage2_Aconst_n=5.dat',
    'StrainRateForage2_Ainv_n=3.dat',]
list_of_stress = [
    'StressForage2_Aconst_n=3.dat',
    'StressForage2_Aconst_n=4.dat',
    'StressForage2_Aconst_n=5.dat',
    'StressForage2_Ainv_n=3.dat']
""" """
gradu = pd.read_csv(path+gradu,delimiter=' ',)#header = ['dudx','dvdx','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz'])
gradu.columns = ['depth','dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz']
graduconst = pd.read_csv(path+graduconst,delimiter=' ',)#header = ['dudx','dvdx','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz'])
graduconst.columns = ['depth','dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz']
strain_uniform = pd.read_csv(path+strain_uniform, delimiter = ' ')
strain_uniform.columns = ['depth','exx','eyy','ezz','exy','eyz','exz']
strain_inv = pd.read_csv(path+strain_inv, delimiter = ' ')
strain_inv.columns = ['depth','exx','eyy','ezz','exy','eyz','exz']
stress = pd.read_csv(path+stress, delimiter = ';')
stress.columns = ['depth','sxx','syy','szz','sxy','syz','sxz']
theta = np.deg2rad(145) # 55°
for p in [gradu, strain_inv, strain_uniform, stress]:
    p['depth'] = -1*p['depth']
    p.set_index('depth',inplace=True)
    rotate_tensor(p, theta)
    
""" Rinse and repeat """
for doc in list_of_gradvit + list_of_strain + list_of_stress:
    if 'Stress' in doc:
        delimiter = ';'
        col = stresscols
    elif 'Strain' in doc:
        delimiter = ' '
        col = straincols
    elif 'GradVit' in doc:
        delimiter = ' '
        col = gradvitcols
    df = pd.read_csv(path + doc , delimiter = delimiter)
    df.columns = col
    if df['depth'].iloc[2] > 0:
        df['depth'] = -1*df['depth']
    #
    df.set_index('depth',inplace=True)
    rotate_tensor(df, theta)
    if 'Stress' in doc:
        """ Effective stress """
        effective_stress = np.sqrt(0.5*(df['sxx']**2 + df['syy']**2 + df['szz']**2) + df['sxz']**2 + df['sxy']**2 + df['syz']**2)
        df['sE'] = effective_stress
    df.to_csv(path+doc[:-4]+'_rotated.dat')
    
#
""" Plot
fig,ax = plt.subplots(figsize = FX.give_dudzsize())
for i in gradu.columns[1:]:
    ax.plot(gradu[i],gradu['depth'],label=i)
ax.grid()
ax.legend()
"""
#
graduconst.to_csv(path+'GradVitForage2_vicoconst_rotated.dat')
gradu.to_csv(path+'GradVitForage2_viscoinv_rotated.dat')
