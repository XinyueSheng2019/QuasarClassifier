



def calculate_obs_SF(mjd, flux):
    # structure function
    
    # convert mjd to integer
    mjd = mjd.astype(int).tolist()
    flux = flux.astype(float).tolist()
    #initialize the delta time
    delta = 1
    obs_SF_list = []
    delta_list = []
    obs_len = max(mjd) - min(mjd)
    while delta < obs_len:
        n = min(mjd)
        count = 0
        mag_vals = 0.0
        while n <= max(mjd)-delta:
            if n in mjd and n+delta in mjd: 
                mag_vals += (flux[mjd.index(n+delta)] - flux[mjd.index(n)])**2   
                count +=1
            n = n + delta
        if count>0:
            obs_SF_list.append(np.sqrt(mag_vals/count))
            delta_list.append(delta)  
        delta +=1
    SF_inft = np.sqrt(2*np.var(flux))
    plt.figure(figsize = (20,10))
    plt.plot(delta_list, obs_SF_list, label = r'$SF(\Delta t)$')
    plt.hlines(SF_inft, min(delta_list), max(delta_list), label = r'$SF_\infty$', color = 'r',linestyle='dashed')
    plt.xlabel('Time lag', fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize = 20)
    plt.ylabel('Structure Function (Observed)', fontsize=20)
    return obs_SF_list, delta_list


def calculate_true_SF(obs_SF_list, delta_list, flux, flux_err = []):
    true_SF_list = []
    if len(flux_err)>0:   
        var_noise = np.var(flux_err)
        print(var_noise)
    else:
        var_noise = 0
    for i in obs_SF_list:
        true_SF_list.append(np.sqrt(i**2 - 2*var_noise))
    SF_inft = np.sqrt(2*np.var(flux))
    plt.figure(figsize = (20,10))
    plt.plot(delta_list, true_SF_list,label = r'$SF(\Delta t)$')
    plt.hlines(SF_inft, min(delta_list), max(delta_list), label = r'$SF_\infty$', color = 'r',linestyle='dashed')
    plt.xlabel('Time lag', fontsize=20)
    plt.legend(fontsize = 20)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Structure Function (True)', fontsize=20)

def caculate_ACF( obs_SF_list, delta_list,flux, flux_err = []):
    #autocorrelation function ACF
    ACF = []
    if len(flux_err)>0:   
        var_noise = np.var(flux_err)
    else:
        var_noise = 0
    var_signal = np.var(flux)
    for i in obs_SF_list:
        ACF.append(1 + (2*var_noise - i**2)/(2*var_signal))
    
    plt.figure(figsize = (20,10))
    plt.plot(delta_list, ACF)
    plt.xlabel('Time lag', fontsize=20)
#     plt.xscale('log')
#     plt.yscale('log')
    plt.ylabel('ACF', fontsize=20)  

def plot_PSD(mag):
    #power spectrum density
    from scipy import signal
    freqs, times, spectrogram = signal.spectrogram(mag)
    plt.figure(figsize = (20,10))
    plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
    plt.title('Spectrogram', fontsize=20)
    plt.ylabel('Frequency band', fontsize=20)
    plt.xlabel('Time window', fontsize=20)
    plt.tight_layout()
    
    freqs, psd = signal.welch(mag)

    plt.figure(figsize = (20,10))
    plt.semilogx(freqs, psd)
    plt.title('PSD: power spectral density', fontsize=20)
    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Power', fontsize=20)
    plt.tight_layout()
