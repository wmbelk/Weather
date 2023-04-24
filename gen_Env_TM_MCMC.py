# %% [markdown]
# # Try out Bayesian update to environmental estimate

# %%
#%%
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import xarray as xr
import arviz as az
import arviz.labels as azl
from hierarchical_normal_belk import hierarchical_normal
import itertools
#!! conda install -c conda-forge flox
import flox
from flox.xarray import xarray_reduce # useful in doing multiple coord groupings

#%%

# %%
rng=np.random.Generator(np.random.PCG64(1234))
#%%
size = 160
mean_tempC_Km = 6.5/1000
max_alt_Km = 13
#keep lat and long square for ease of matrixing
horz_offest = 10
lat = np.arange(horz_offest, size)
long = np.arange(0, size - horz_offest)
alt = np.arange(0, max_alt_Km)*1000
#%%

# %%
def sample_AR_signal(n_samples, corr, mu=0, sigma=1):
    assert 0 < corr < 1, "Auto-correlation must be between 0 and 1"
    burn_samples = 100
    n_samples=n_samples+burn_samples

    # Find out the offset `c` and the std of the white noise `sigma_e`
    # that produce a signal with the desired mean and variance.
    # See https://en.wikipedia.org/wiki/Autoregressive_model
    # under section "Example: An AR(1) process".
    c = mu * (1 - corr)
    sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

    # Sample the auto-regressive process.
    signal = [c + np.random.normal(0, sigma_e)]
    for _ in range(1, n_samples):
        signal.append(c + corr * signal[-1] + np.random.normal(0, sigma_e))
    #signal=signal[burn_samples:n_samples]

    return np.array(signal[burn_samples:])

def compute_corr_lag_1(signal):
    return np.corrcoef(signal[:-1], signal[1:])[0][1]
#%%

# %% [markdown]
# 
# Baseline thermal along latitude

# %%
base_sigma = .05
samp_lat_base = sample_AR_signal(size-horz_offest, 0.5, mu=2, sigma=base_sigma)
samp_lat= pd.DataFrame(samp_lat_base)
print(compute_corr_lag_1(samp_lat_base), samp_lat)#.iloc[:,0]),compute_corr_lag_1(samp_lat.iloc[0,:]))
# %%

# %% [markdown]
# Extend along longitude

# %%
samp = sample_AR_signal(size-horz_offest, 0.75, mu=samp_lat, sigma=base_sigma)
samp = pd.DataFrame(samp[:, :, 0])
print(compute_corr_lag_1(samp.iloc[:,0]),compute_corr_lag_1(samp.iloc[0,:]))
# %%

# %%
def plot_temperature_env(samp):
    x2, y2 = np.meshgrid(samp.index.values, samp.columns.values)
    plt.figure(figsize=(6,5))
    axes = plt.axes(projection='3d')
    axes.plot_surface(x2, y2,samp.values,cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    axes.set_ylabel('Longitude')
    axes.set_xlabel('Latitude')
    axes.set_zlabel('Temperature')
    # keeps padding between figure elements
    plt.tight_layout()
    plt.show()

plot_temperature_env(samp)
# %%

# %% [markdown]
# Add trend on top of the AR variation -- to baseline thermal

# %%
lat_inc_mu = 15/100
lat_inc_max = lat_inc_mu *150
long_inc_mu, long_inc_std = 25/100, .1

def add_inc_MA(size, horz_offest, sample_AR_signal, samp_lat, lat_inc_max, long_inc_mu, long_inc_std):
    lat_inc = np.linspace(0,lat_inc_max, len(samp_lat))
    sample_lat_inc = samp_lat[0] + lat_inc
    sample_lat_inc = pd.DataFrame(sample_lat_inc)
#sample_lat_inc.plot()

    samp_inc = sample_AR_signal(size-horz_offest, corr=0.5, mu=sample_lat_inc)
    long_inc = stats.norm.rvs(loc=long_inc_mu, scale=long_inc_std, size=(size-horz_offest,size-horz_offest), random_state=None)
    long_inc = np.cumsum(long_inc, axis=0)
    samp_inc = pd.DataFrame(samp_inc[:, :, 0]+long_inc)
    return samp_inc

samp_inc = add_inc_MA(size, horz_offest, sample_AR_signal, samp_lat, lat_inc_max, long_inc_mu, long_inc_std)


plot_temperature_env(samp_inc)
# %%

# %% [markdown]
# Extend into atmosphere

# %%
#allow for inversion by having random lapse rate at diff altitudes
def add_altitude_effects(rng, samp_inc, mean_tempC_Km, max_alt_Km):
    tempC_Km = rng.normal(loc=mean_tempC_Km, scale=mean_tempC_Km/10, size=max_alt_Km)
# Temp at altitude = base temp - tempC_km * altitude
    temperature = ( [np.array(samp_inc) 
                 for _ in np.arange(max_alt_Km)]
               -np.broadcast_to(
    tempC_Km * np.arange(max_alt_Km)*1000, (size-horz_offest,size-horz_offest,max_alt_Km)
    ).T
)
    temperature = temperature.T
    return temperature

temp_3D = add_altitude_effects(rng, samp_inc, mean_tempC_Km, max_alt_Km)
# %%

# %%
xr_temp_3D = xr.DataArray(temp_3D, dims=['lat', 'long', 'alt'], coords={'lat': lat, 'long': long, 'alt': alt})
fig = xr_temp_3D.plot.contourf(x='lat',y='long',col='alt', col_wrap=4,
                         robust=True, vmin=-90, vmax=32, levels=20)
plt.suptitle('Temperature at different altitudes', fontsize = 'xx-large',
             weight = 'extra bold')
plt.subplots_adjust(top=.92, right=.8, left=.05, bottom=.05)

xr_tempC_Km=  xr.DataArray(mean_tempC_Km, dims=['alt'], coords={'alt': alt})

# %% [markdown]
# Calculate pressure based on baseline temp field and assumed L; 
# 

# %%
# %%
#barometric formula
def add_barometric_effects(T = 288.15-273.15, L = 0.0065, H = 0,  P0 = 101_325.00, g0 = 9.80665, M = 0.0289644, R = 8.3144598):
    #barometric formula
    #P = P0 * (1 - L * H / T0) ^ (g0 * M / (R * L))
    #P = pressure
    #P0 = pressure at sea level = 101_325.00 Pa
    #L = temperature lapse rate = temperature lapse rate (K/m) in
    #H = altitude (m)
    #T0 = temperature at sea level = reference temperature (K)
    #g0 = gravitational acceleration = gravitational acceleration: 9.80665 m/s2
    #M = molar mass of air = molar mass of Earth's air: 0.0289644 kg/mol
    #R = gas constant = universal gas constant: 8.3144598 J/(mol·K)
    #L = temperature lapse rate
    #T = temperature
    T = T +273.15
    if isinstance(T, xr.core.dataarray.DataArray):
        T0 = T.sel(alt=0)
        
    else:
        T0 = T[0]
        print('used t[0]')
        print(type(T))
    #return P0 * (1 - L * H / (T0+273.15)) ** (g0 * M / (R * L))
    return P0 * (T / T0) ** (g0 * M / (R * L.mean()))


pressure = add_barometric_effects(T = xr_temp_3D, 
                                 L = xr_tempC_Km, 
                                 H = xr_temp_3D.alt,  P0 = 101_325.00, g0 = 9.80665, M = 0.0289644, R = 8.3144598)
   

# %%
pressure

# %%
# %%
xr_temp_pres = xr.merge(
    [xr_temp_3D.rename("Temperature"), 
     pressure.rename("Pressure")]
     )
# %%
xr_temp_pres.Pressure.plot.contourf(x='lat',y='long', col='alt', col_wrap=4,
                         robust=True, levels=20)
plt.suptitle('Pressure at different altitudes', fontsize = 'xx-large',
             weight = 'extra bold')
plt.subplots_adjust(top=.92, right=.8, left=.05, bottom=.05)
# %%

# %% [markdown]
# # make trajectory and get corresponding temp and pres

# %%
# %%
# make Z = a function of time and  X = sin of time and y = cos of time
time = pd.to_datetime( np.arange(0, 120*60, 1), unit='s')
print(time)

# %%
release_alt = 12_000 #Troposphere goes to about 12Km, thermal is about linear there
step_alt = 1

turn_rate = 2
x = (np.sin((time.minute/60 +time.second)*2*np.pi*turn_rate) +1) * size/2 +30
y = (np.cos((time.minute/60 +time.second)*2*np.pi*turn_rate) +1 ) * size/2
#create samples from normal distribution and sort them
samples = stats.weibull_max.rvs(1.5, loc=0, scale=1, size=len(time), random_state=None)
samples.sort()
steps = samples/(samples.max()-samples.min()) /1.3  #normalize and shrink
steps = steps - steps.min() #shift to 0
 #smaller step per time
z = release_alt * (1- steps)

plt.plot(time, z)
plt.xlabel('Time')
plt.ylabel('Altitude')
plt.title('Altitude vs Time')
ax = plt.gca()
ax.set_ylim(0, 12000)
plt.show()
#plot 3d trajectory of z by x and y
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_zlim(0, 12000)
plt.title('3D Trajectory')
plt.show()
# %%

# %%
#select from xarray the temperature at the pressure of the trajectory
xr_x = xr.DataArray(x, dims=['time'], coords={'time': time})
xr_y = xr.DataArray(y, dims=['time'], coords={'time': time})
xr_z = xr.DataArray(z, dims=['time'], coords={'time': time})

xr_traj_env = xr_temp_pres.interp(lat=xr_x,long=xr_y,alt=xr_z)#, method='nearest')
xr_traj_env = xr_traj_env.interpolate_na(dim='time', method='linear', fill_value="extrapolate")
xr_traj_env.attrs =dict(units='seconds since 1970-01-01 00:00:00')
 # delay start of trajectory
xr_traj_env['time'] = xr_traj_env.time +pd.Timedelta(hours=.75)

xr_traj_env
# put in ballon data before resampling

#%% [markdown]
#Put in Ballon data

#%%

ballon_alt_samples = np.arange(start=0,stop=max_alt_Km*1000+1,step=500)
ballon_time = ballon_alt_samples/5
ballon_time = pd.to_datetime(  ballon_time, unit='s')
ballon_lat = [40, 100]
ballon_long = [40, 125]
launch_count = 2
ballon_delay = pd.Timedelta(hours=7)# 7*60*60 # 7 hrs later in seconds
launch_idx = np.arange(0,launch_count)
def ballon_release(xr_temp_pres, ballon_alt_samples, ballon_time, ballon_lat, ballon_long, ballon_delay, launch_idx):
    #ballon launch delay is in hours, will convert number to pd.Timedelta
    ballon_delay = pd.Timedelta(hours=ballon_delay)
    coords={'launch':[launch_idx],'time':(('time'),ballon_time+ballon_delay)}
    xr_ballon_env = xr_temp_pres.interp(lat=
                                    xr.DataArray([[ballon_lat[launch_idx]]]*len(ballon_time), 
                                                 dims=['time','launch'],
                                                 coords=coords),
                                    long=
                                    xr.DataArray([[ballon_long[launch_idx]]]*len(ballon_time),
                                                 dims=['time','launch'],
                                                 coords=coords),
                                    alt=
                                    xr.DataArray([ballon_alt_samples],
                                                 dims=['launch','time'],
                                                 coords=coords),
                                    )
    xr_ballon_env= xr_ballon_env.interpolate_na(dim='time',method='linear', fill_value = 'extrapolate')
    xr_ballon_env.attrs =dict(units='seconds since 1970-01-01 00:00:00')
    return xr_ballon_env


xr_ballon_env_0 = ballon_release(xr_temp_pres, ballon_alt_samples, ballon_time, ballon_lat, ballon_long, ballon_delay=0, launch_idx=0)
xr_ballon_env_1 = ballon_release(xr_temp_pres, ballon_alt_samples, ballon_time, ballon_lat, ballon_long, ballon_delay=7, launch_idx=1)

#TODO: remove the 'launch' dimension from the ballon_release function, then do not squeeze it out
xr_ballon_env = xr.concat([xr_ballon_env_0.squeeze(), 
                           xr_ballon_env_1.squeeze()], 
                           dim='time')
xr_ballon_env

xr_traj_env = xr.concat([xr_traj_env, 
                         xr_ballon_env.drop('launch')],
                           dim='time').sortby('time')
xr_traj_env

'''# %%
xr_traj_env.resample(time='5min').mean().Temperature.plot()
xr_traj_env.resample(time='10min').mean().Temperature.plot()'''

# %%
xr_traj_env.Temperature.plot()
plt.suptitle('Temperature over time', fontsize = 'xx-large')
plt.show()
xr_traj_env.Pressure.plot()
plt.suptitle('pressure over time', fontsize = 'xx-large')
plt.show()

# %%
#add wind direction and speed then its velocity relvant to the trajectory
wind_direction = 180 #degrees from north is 0, from east is 90, from south is 180, from west is 270
wind_speed = 10 #m/s #TODO: add wind speed as a function of altitude and lat long
wind_speed_long = wind_speed * np.cos(np.deg2rad(wind_direction))
wind_speed_lat = wind_speed * np.sin(np.deg2rad(wind_direction))
wind_speed_z = 0
print('wind_speed_long',wind_speed_long, 
        'wind_speed_lat',wind_speed_lat, 
        'wind_speed_z',wind_speed_z)
# %%

# %%
#add wind velocity relevant to the trajectory; dont add wind in
if False:
    xr_traj_env['wind_speed_long'] = wind_speed_long
    xr_traj_env['wind_speed_lat'] = wind_speed_lat
    xr_traj_env['wind_speed_z'] = wind_speed_z
    xr_traj_env['wind_speed'] = np.sqrt(wind_speed_long**2 + wind_speed_lat**2 + wind_speed_z**2)
    xr_traj_env['wind_direction'] = wind_direction
    xr_traj_env=xr_traj_env.interpolate_na(dim='time', method='linear', limit=None, use_coordinate=True, fill_value='extrapolate')
    xr_traj_env
# %%

# %% [markdown]
# # Using Xarray resampling

# %%
xr_traj_env

# %%
#downsample from xarray
#must be a datetime index in xarray
# move xarray coordinate to variable
xr_traj_env_time = xr_traj_env.reset_coords(['lat','long','alt'], drop=False)
xr_traj_env_time = xr_traj_env_time.resample(time='5min', restore_coord_dims=True).mean().dropna(dim='time')
xr_traj_env_time_coords = xr_traj_env_time
#Move variable to xarray coordinate
xr_traj_env_time = xr_traj_env_time.drop(['lat','long','alt'])
xr_traj_env_time = xr_traj_env_time.expand_dims({"lat":xr_traj_env_time_coords.lat.values, 
                              'long':xr_traj_env_time_coords.long.values, 
                              'alt':xr_traj_env_time_coords.alt.values}) 


# %%
# toDO CHANGE THIS VARIABLE BELOW HERE
#xarray make a multiindex of lat long alt and time

#grp_traj_env = 
# may be useful : xr_traj_env_time.stack(alt_lat_long_time=['alt','lat','long','time'],create_index=True)



# %% [markdown]
# # Using average values per Km; TODO: find more principled way to remove autocorrelation 

# %%
'''bins_alt = np.linspace(alt.min(), alt.max(), 11)
bins_lat =[lat.min(), lat.mean(),lat.max()] #quadrents
bins_long = [long.min(), long.mean(), long.max()]# quadrents
bins_time = np.arange(time.min(), time.max()+1, 10)'''

# %%
#grouping lat long and alt
if False:
  grp_traj_env=xarray_reduce(xr_traj_env.drop_vars(['wind_speed_long', 'wind_speed_lat', 'wind_speed_z', 'wind_speed', 'wind_direction']),
               'alt', 'lat', 'long',
                 func='mean',
                 expected_groups=(
                            pd.IntervalIndex.from_breaks(bins_alt, closed='left'),
                            pd.IntervalIndex.from_breaks(bins_lat, closed='left'),
                            pd.IntervalIndex.from_breaks(bins_long, closed='left')
                        ))
  grp_traj_env

# %%
#grouping lat long and alt and time
'''grp_traj_env=xarray_reduce(xr_traj_env, #.drop_vars(['wind_speed_long', 'wind_speed_lat', 'wind_speed_z', 'wind_speed', 'wind_direction']),
               'alt', 'lat', 'long', 'time',
                 func='mean',
                 expected_groups=(
                            pd.IntervalIndex.from_breaks(bins_alt, closed='left'),
                            pd.IntervalIndex.from_breaks(bins_lat, closed='left'),
                            pd.IntervalIndex.from_breaks(bins_long, closed='left'),
                            pd.IntervalIndex.from_breaks(bins_time, closed='left')
                        ))

grp_traj_env = grp_traj_env.stack(alt_lat_long_time=(
    'alt_bins', 
    'lat_bins', 
    'long_bins',
    'time_bins')).dropna(dim='alt_lat_long_time')

grp_traj_env.coords'''

# %% [markdown]
# # Model temp and pressure varying by altitude, lat, & long

# %%

coords={'alt_lat_long_time':
                      np.arange(xr_traj_env_time.sizes['time'], dtype=int)
                      }
coords

# %%
with pm.Model(coords=coords) as thermal_pres:
    #Temp is in celcius
    
    Alt_ = pm.ConstantData('Altitude_m', xr_traj_env_time_coords.alt.values,#[bin_item.mid for bin_item in grp_traj_env.alt_bins.values], 
                                          dims='alt_lat_long_time' )
    Lat_ = pm.ConstantData('Latitude', xr_traj_env_time_coords.lat.values,# [bin_item.mid for bin_item in grp_traj_env.lat_bins.values],
                                        dims='alt_lat_long_time' )
    Long_ = pm.ConstantData('Longitude', xr_traj_env_time_coords.long.values,#[bin_item.mid for bin_item in grp_traj_env.long_bins.values],
                                          dims='alt_lat_long_time' )
    Temp_ = pm.ConstantData('Temperature_Samples', xr_traj_env_time_coords.Temperature.values, dims='alt_lat_long_time' )
    Pres_ = pm.ConstantData('Pressure_Samples', xr_traj_env_time_coords.Pressure.values, dims='alt_lat_long_time' )
    #prior on effect on temp (degC) of altitude and lat, long
    baseline_temp = pm.Normal('baseline_temp', mu=0, sigma=5) #'L'
    Alt_effect_temp = pm.Normal('Alt_effect_temp_Km', mu=-6, sigma=.5)
    Lat_effect_temp = pm.Normal('Lat_effect_temp', mu=0, sigma=.01)
    Long_effect_temp = pm.Normal('Long_effect_temp', mu=0, sigma=.01)
    #prior on temp and pressure
    #TODO: PULL FROM DATABASE into a pm.Interpolated...maybe not: need relationship between data spreads?
    mu_t = pm.Deterministic('mu_t',
                               baseline_temp + Alt_effect_temp/1000 * Alt_ + Lat_effect_temp * Lat_ + Long_effect_temp * Long_, 
                               dims='alt_lat_long_time')
    #mu_t = hierarchical_normal('temperature_mean', mu= mu_mu_t, sigma = 2, dims='alt_lat_long_time')
    #mu_p = hierarchical_normal('pressure_mean', 
    P0 = Pres_[0]#101_325.00
    g0 = 9.80665
    M = 0.0289644
    R = 8.3144598

    mu_p= pm.Deterministic('mu_p',P0 *  ((mu_t+273.15)/(Temp_[0]+273.15)) ** (g0 * M / (R * (-Alt_effect_temp/1000))), #needed negative b/c the lapse is positive, but use addition in effect
                                 dims='alt_lat_long_time')
    '''add_barometric_effects(T = mu_t,#Temp_, 
                                 L = Alt_effect_temp/1000, H = Alt_,  
                                 P0 = 101_325.00, g0 = 9.80665, M = 0.0289644, R = 8.3144598)'''
    #add_barometric_effects = P0 * (T/T0) ** (g0 * M / (R * L))
    #prior on error variation
    sigma_t=pm.Exponential('model_error_t', 1/15)
    sigma_p=pm.Exponential('model_error_p', 1/5000)
    #adjusted temp - normal dist error term
    obs_t = pm.Normal('obs_t', mu=mu_t, sigma=sigma_t, 
                    observed = Temp_, dims='alt_lat_long_time')
    obs_p = pm.Normal('obs_p', mu=mu_p, sigma=sigma_p, 
                    observed = Pres_, dims='alt_lat_long_time')
    
pm.model_to_graphviz(thermal_pres)

# %%
with thermal_pres:
    idata2 = pm.sample_prior_predictive(1000)
az.plot_ppc(idata2, group='prior', kind='cumulative')

# %%
with thermal_pres:
    idata2.extend(pm.sample(1000, tune=1000, chains = 4, cores=1))

    az.plot_trace(idata2)
    plt.subplots_adjust (hspace=0.4)#, wspace=0.4) 
    

# %%
#xarray filter by values
lat_min = idata2.constant_data.Latitude.min()
lat_max = idata2.constant_data.Latitude.max()
long_min = idata2.constant_data.Longitude.min()
long_max = idata2.constant_data.Longitude.max()
lat_mid = (lat_min + lat_max)/2
long_mid = (long_min + long_max)/2


idx_north = idata2.constant_data.where(idata2.constant_data.Latitude>lat_mid, drop=True).alt_lat_long_time.values
idx_south = idata2.constant_data.where(idata2.constant_data.Latitude<lat_mid, drop=True).alt_lat_long_time.values
idx_east = idata2.constant_data.where(idata2.constant_data.Longitude>long_mid, drop=True).alt_lat_long_time.values
idx_west = idata2.constant_data.where(idata2.constant_data.Longitude<long_mid, drop=True).alt_lat_long_time.values

# %%

class DimCoordLabeller_alt(azl.BaseLabeller):
    """WIP."""
    def __init__(self, coords_ds):
        self.coords_ds = xr.Dataset(coords)

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """WIP."""#format decimals in f statement
        temp =  self.coords_ds.sel(pointwise_sel=coord_val).items()
        temp = [(v.values) for _,v in temp][0]
        return f"{temp:.2f}" 
    
coords = {
    'alt_lat_long_time': xr.DataArray(
        idata2.constant_data.Altitude_m.values, 
        dims=['pointwise_sel'],coords={'pointwise_sel': idata2.constant_data.alt_lat_long_time.values}
    )
}        


labeller = DimCoordLabeller_alt(coords)

# %%

#figures with lat in coulmns and long in rows
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()
for i, ((N_S_label,N_S_idx), (E_W_label,E_W_idx)) in enumerate([[i,j] 
                                  for i in [('North',idx_north),('South',idx_south)] 
                                  for j in [('West',idx_west), ('East',idx_east)]]):
    print(N_S_label,E_W_label)
    ax[i].set_title(f'Lat: {N_S_label} Long: {E_W_label}')
    idx = np.intersect1d(N_S_idx,E_W_idx)
    az.plot_forest(idata2.sel(alt_lat_long_time=idx), 
                   var_names=['mu_t'],
                   kind='ridgeplot', 
                   combined=True, ax= ax[i],
                   labeller=labeller
                   )
    #align the y axis
    #ax[i].set_ylim(0, 10000)
    ax[i].set_xlim(-70, 10)
    ax[i].grid()


# %%
az.plot_forest(idata2, var_names=['mu_t'],kind='ridgeplot', combined=True)#,combine_dims='time_bins')
az.plot_forest(idata2, var_names=['mu_p'],kind='ridgeplot', combined=True)

# %%
with thermal_pres:
    # pymc sample posterior predictive check
    pm.sample_posterior_predictive(idata2, extend_inferencedata=True)
    az.plot_ppc(idata2, group='posterior', kind='cumulative')


# %%
az.plot_dist_comparison(idata2, kind='observed', labeller=labeller)
