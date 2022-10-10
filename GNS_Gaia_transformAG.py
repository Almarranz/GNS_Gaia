#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable
from skimage.transform import estimate_transform
from astropy.io import fits
from astropy.wcs import WCS


# In[2]:


#stars_calibrated_H_chip2.txt 
#Gaia edr3 -> dr3
GNS2 =  '/Users/amartinez/Desktop/PhD/HAWK/GNS_2/quality_check/GNS_lists/'
field = 6
tab = Table.read(GNS2 + 'Field%s/XMatch_01.xml'%(field), format='votable')
col_nom = tab.colnames
print(tab.colnames)

print(tab['pmra_error'][0:5])
# In[3]:


header = fits.getheader(GNS2 + '/Field%s/%s_chip1_holo_cal.fits.gz'%(field, field))
w = WCS(header)
header


# In[4]:


# selection = np.argwhere((tab['PM_tab1'] > 0) & (tab['e_pmRA_tab1'] < 0.3) & (tab['e_pmDE_tab1'] < 0.3))
selection = np.argwhere((tab['pm'] > 0) & (tab['pmra_error'] < 0.3) & (tab['pmdec_error'] < 0.3))

tab = tab[selection]
print(len(tab))


# In[5]:


# gnstime = Time(['2021-09-23T00:00:00'], format='isot')
if field == 6:
    gnstime = Time(['2022-05-27T00:00:00'], format='isot')
if field == 9:
    gnstime = Time(['2021-09-23T00:00:00'], format='isot')
    
gnsequinox = Time(['2000-01-01T12:00:00'], format='isot')

GNSCoord = SkyCoord(ra=tab['col1'],
                   dec=tab['col2'], 
                   frame='icrs', unit='deg',
                    equinox=gnsequinox,
                   obstime=gnstime)

#GNSCoord = SkyCoord(ra=tab['_RAJ2000_tab1'],
#                   dec=tab['_DEJ2000_tab1'], 
#                   frame='icrs', unit='deg',
#                    equinox=gnsequinox,
#                   obstime=gnstime)


# In[6]:


gaiaequinox = Time(['2000-01-01T12:00:00'], format='isot')
#gaiaequinox = Time(['2016-01-01T12:00:00'], format='isot')
GaiaCoord = SkyCoord(ra=tab['ra'],
                   dec=tab['dec'],
                    pm_ra_cosdec=tab['pmra'],
                     pm_dec=tab['pmdec'],
                   frame='icrs',
                     equinox = gaiaequinox,
                    obstime=Time(['2016-01-01T00:00:00'], format='isot'))

#GaiaCoord.ra.deg[0:5],tab['RA_ICRS_tab1'][0:5]


# In[7]:

if field == 6:
    t = Time(['2022-05-27T00:00:00','2016-01-01T00:00:00'],scale='utc')
if field == 9:
    t = Time(['2021-09-23T00:00:00','2016-01-01T00:00:00'],scale='utc')
delta_t = t[0] - t[1]
#delta_t
#<TimeDelta object: scale='tai' format='jd' value=2092.000011574074>
#2092.000011574074 differenza in giorni che corrisponde a circa 5 anni.


# In[71]:


#GNSCoord (_RAJ2000_tab1, _DEJ2000_tab1)
#<SkyCoord (ICRS): (ra, dec) in deg
#    [[(266.41621043, -28.98607206)],
#     [(266.41319531, -28.98148708)],
#     [(266.40523322, -28.98019937)],
#     [(266.42215027, -28.97921259)],
#     [(266.40752751, -28.97885245)],

#GNSCoord (C1_tab2, C2_tab2)
#<SkyCoord (ICRS): (ra, dec) in deg
#    [[(266.416199, -28.986073)],
#     [(266.413177, -28.981491)],
#     [(266.405243, -28.980185)],
#     [(266.42215 , -28.97921 )],
#     [(266.407532, -28.978851)],


# In[72]:


#GaiaCoord (RA_ICRS_tab1, DE_ICRS_tab1)
#<SkyCoord (ICRS): (ra, dec) in deg
#    [[(266.41620808, -28.98608851)],
#     [(266.41319059, -28.98150535)],
#     [(266.40523032, -28.98020685)],
#     [(266.42215357, -28.97922212)],
#     [(266.40752028, -28.97886865)],

#     (266.41620808, -28.98608851)],
#     [(266.41319059, -28.98150535)],
#     [(266.40523032, -28.98020685)],
#     [(266.42215357, -28.97922212)],
#     [(266.40752028, -28.97886865)],


# In[8]:


GaiaGNSCoord=GaiaCoord.apply_space_motion(dt=delta_t)
#GaiaGNSCoord
#<SkyCoord (ICRS): (ra, dec) in deg
#    [[(266.41620723, -28.9860944 )],
#     [(266.41318889, -28.98151189)],
#     [(266.40522928, -28.98020953)],
#     [(266.42215475, -28.97922554)],
#     [(266.40751769, -28.97887445)],

# array([[-3.19456939e-06],
#        [-6.41613241e-06],
#        [-3.94641387e-06],
#        [ 4.47739518e-06],
#        [-9.81071412e-06]]))
print(GaiaGNSCoord.ra.deg[0:5],GaiaCoord.ra.deg[0:5],GaiaGNSCoord.ra.deg[0:5]-GaiaCoord.ra.deg[0:5])


# FROM COORDINATE TO PIXELS

# In[9]:


#xgaia, ygaia = w.world_to_pixel(GaiaCoord)
#Pixel Gaia con coordinate di Gaia corrette al tempo di GNS (usando i valori di moto proprio di Gaia)
xgaia, ygaia = w.world_to_pixel(GaiaGNSCoord)
n_sources = len(xgaia)
xgaia = np.ndarray.flatten(xgaia)
ygaia = np.ndarray.flatten(ygaia)
print(xgaia[0:5], ygaia[0:5])
#array([5018.10455091, 4845.71579371, 5015.10482536, 4437.73514851,
#        4867.47478829]),
# array([ 864.00894398, 1177.60745028, 1626.23979519,  803.94862539,
#        1557.25207911]))


# In[10]:


#Pixel corrispondenti alle coordinate GNS epoca de observacion 2021 
xGNS, yGNS = w.world_to_pixel(GNSCoord)
#print(xGNS)
xGNS = np.ndarray.flatten(xGNS)
yGNS = np.ndarray.flatten(yGNS)
xoriGNS = np.ndarray.flatten(np.transpose(np.array(tab['col3'])))
yoriGNS = np.ndarray.flatten(np.transpose(np.array(tab['col4'])))
#
print(xGNS[0:5],xoriGNS[0:5], (xGNS-xoriGNS)[0:5])


# In[15]:


src = np.zeros((n_sources,2))
src[:,0] = xgaia
src[:,1] = ygaia
dst = np.zeros((n_sources,2))
dst[:,0] = xGNS
dst[:,1] = yGNS
dst1 = np.zeros((n_sources,2))
dst1[:,0] = xoriGNS
dst1[:,1] = yoriGNS
trans = estimate_transform('polynomial',src,dst,order=1)
#questo é il corretto; order =1 corrisponde a GNS_Gaia_1ori.pdd
#                    ; order =2 corrisponde a GNS_Gaia_2ori.pdd 
trans1 = estimate_transform('polynomial',src,dst1,order=2) 

print((src)[0:5])
print(trans.params)
print(trans(src)[0:5])
print((dst)[0:5])

print(trans1.params)
print(trans1(src)[0:5])
print((dst1)[0:5])

#print(trans(dst)[0:5])


# In[115]:


print(trans.params[0,:])
print(trans.params[1,:])


# In[116]:


NGaiainGNS = trans(src)
xNGaiainGNS = NGaiainGNS[:,0]
yNGaiainGNS = NGaiainGNS[:,1]


# In[117]:


pix_scale = 0.053
dx = (xNGaiainGNS -xGNS) * pix_scale
dy = (yNGaiainGNS -yGNS) * pix_scale
x_med = np.median(dx)
x_std = np.std(dx)
print(f"Median offset: {x_med}")
print(f"Standard deviation: {x_std}")
y_med = np.median(dy)
y_std = np.std(dy)
print(f"Median offset: {y_med}")
print(f"Standard deviation: {y_std}")


# In[17]:


#questo é il corretto
NGaiainGNS1 = trans1(src)
xNGaiainGNS1 = NGaiainGNS1[:,0]
yNGaiainGNS1 = NGaiainGNS1[:,1]


# In[18]:


#questo é il corretto
pix_scale = 0.053
dx1 = (xNGaiainGNS1 -xoriGNS) * pix_scale
dy1 = (yNGaiainGNS1 -yoriGNS) * pix_scale
x1_med = np.median(dx1)
x1_std = np.std(dx1)
print(f"Median offset: {x1_med}")
print(f"Standard deviation: {x1_std}")
y1_med = np.median(dy1)
y1_std = np.std(dy1)
print(f"Median offset: {y1_med}")
print(f"Standard deviation: {y1_std}")


# In[52]:


#pix_scale = 0.053
#dx = (xgaia - xGNS_in_Gaia) * pix_scale
#dy = (ygaia - yGNS_in_Gaia) * pix_scale
#x_med = np.median(dx)
#y_std = np.std(dx)
#print(f"Median offset: {x_med}")
#print(f"Standard deviation: {x_std}")
#y_med = np.median(dy)
#y_std = np.std(dy)
#print(f"Median offset: {y_med}")
#print(f"Standard deviation: {y_std}")


# In[53]:


print(0.4 * 0.053)


# In[121]:


plt.rc('font', size = 16)
fig, axes = plt.subplots(figsize=(10,10))
axes.set_title('Field%s'%(field))
axes.hist(dy,bins=12,label='Declination')
axes.hist(dx,bins=12,label='Right Ascension',alpha=0.5)
# axes.set_xlim(-0.12,0.12)
axes.set_xlabel('Offset in mas')
axes.set_ylabel('Number of stars')
axes.legend()


# In[55]:


# fig.savefig('GNS_Gaia_2.pdf',dpi='figure', format='pdf')


# In[14]:


#questo é il corretto
plt.rc('font', size = 16)
fig, axes = plt.subplots(figsize=(10,10))
axes.set_title('Field%s'%(field))
axes.hist(dy1,bins=12,label='Declination')
axes.hist(dx1,bins=12,label='Right Ascension',alpha=0.5)
# axes.set_xlim(-0.12,0.12)
axes.set_xlabel('Offset in mas')
axes.set_ylabel('Number of stars')
axes.legend()


# In[133]:


fig.savefig('GNS_Gaia_1ori.pdf',dpi='figure', format='pdf')


# In[ ]:




