import numpy as np
from astropy.io import fits
from astropy.modeling import models
from skimage.morphology import label

from scipy.interpolate import griddata
from astropy.convolution import convolve, Box1DKernel

import sys
import matplotlib.pyplot as plt

import pylinear

# Noise terms
BCK_ZODIACAL = 1.047  # e/pix/sec
BCK_THERMAL = 0.0637249  # e/pix/sec
DARK = 0.005  # e/pix/sec
READNOISE_RMS = 15
# Effective sky
SKY = BCK_ZODIACAL + BCK_THERMAL


def extract_outside_pylinear():
    # get data
    img = fits.open('simp1_flt.fits')
    dat = img[1].data
    # coords eyeballed from ds9
    cutout = dat[1600:1620, 1770:2670]

    # now collapse the cutout along the spatial direction
    spec_1d = np.sum(cutout, axis=0)

    # get sens curve
    sens = fits.open('Wang2022_sens_0720_2020.fits')

    # create a lambda array
    # this will only match approximately
    lam = np.linspace(9500, 20000, len(spec_1d))

    # get to physical units
    sn = sens[1].data['Sensitivity']
    sn_lam = sens[1].data['Wavelength']
    sens_regrid = griddata(points=sn_lam, values=sn, xi=lam)

    spec_1d_phys = spec_1d / sens_regrid

    # 2D spectrum
    fig = plt.figure(figsize=(8, 1))
    ax = fig.add_subplot(111)
    ax.imshow(cutout, origin='lower')
    fig.savefig('basic_2d_cutout.png', dpi=200)

    # plot sens
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(sn_lam, sn, color='g')
    # ax1.plot(lam, sens_regrid, color='y')
    fig1.savefig('Wang2022_sensitivity.png', dpi=200)

    # 1d spec figure
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(lam, spec_1d_phys, color='k')

    # also plot smoothed version and model
    box = Box1DKernel(width=20)
    smooth_spec = convolve(spec_1d_phys, box)
    ax2.plot(lam, smooth_spec, '--', color='r')

    # model
    model = np.genfromtxt('5329.sed', dtype=None, names=['wav', 'flam'],
                          encoding='ascii', skip_header=3)
    model_lam = model['wav']
    model_spec = model['flam']
    ax2.plot(model_lam, model_spec, color='skyblue')

    ax2.set_xlim(9800, 19500)
    # ax2.set_ylim(1e-19, 1.5e-18)
    fig2.savefig('basic_ex_1d_spec.png', dpi=200)

    return None


def get_counts(mag):
    counts = np.power(10, -0.4 * (mag - ZP))
    return counts


def pylinearnoise(sci, size, exptime):
    signal = (sci + SKY + DARK)
    signal = signal * exptime
    variance = signal + READNOISE_RMS**2
    sigma = np.sqrt(variance)
    new_sig = np.random.normal(loc=signal, scale=sigma, size=size)
    final_sig = (new_sig / exptime) - SKY
    return final_sig


if __name__ == '__main__':

    # Define lst file paths
    sedlst = 'roman_grism_test_sed.lst'
    fltlst = 'roman_grism_test_flt.lst'
    obslst = 'roman_grism_test_obs.lst'
    wcslst = 'roman_grism_test_wcs.lst'

    # Paths and other prep
    dirimage = 'roman_grism_test_dirimg.fits'
    segfile = dirimage.replace('dirimg.fits', 'segmap.fits')
    tablespath = './tables/'

    maglim = 30.0
    beam = '+1'
    NPIX = 4096
    ZP = 26.3
    mag1, mag2 = 19.5, 20.5
    exptime = 3600

    # First create direct image and segfile
    full_img = np.zeros((NPIX, NPIX))
    # Create Gaussian function models
    # stddev in pixels
    gaussfunc1 = models.Gaussian2D(x_stddev=3, y_stddev=3)
    gaussfunc2 = models.Gaussian2D(x_stddev=3, y_stddev=3)

    # Put the two galaxies at predetermined locations
    gaussfunc1.x_mean = 1000
    gaussfunc1.y_mean = 1000
    gaussfunc2.x_mean = 1000
    gaussfunc2.y_mean = 2000

    x, y = np.meshgrid(np.arange(NPIX), np.arange(NPIX))
    full_img += gaussfunc1(x, y)
    full_img += gaussfunc2(x, y)

    # Get required coutns
    counts1 = get_counts(mag1)
    counts2 = get_counts(mag2)
    all_counts = [counts1, counts2]

    # Generate segmap
    threshold = 0.3  # threshold to apply to the image
    good = full_img > threshold  # these pixels belong to a source
    segmap = label(good)  # now these pixels have unique SegIDs

    nonzero_idx = np.where(segmap != 0)
    segids = np.sort(np.unique(segmap[nonzero_idx]))

    for i in range(len(segids)):
        segid = segids[i]
        segidx = np.where(segmap == segid)
        src_counts = np.sum(full_img[segidx])
        counts = all_counts[i]

        # scale
        scale_fac = counts / src_counts

        # Do not multiply the full img by the scale fac
        # or other sources will also be affected
        full_img[segidx] *= scale_fac

        print(segid, src_counts, counts, scale_fac)

    # Save
    # Create a header (pyLINEAR needs WCS)
    # Get a header with WCS
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    img_sim_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'
    dirimg_hdr = fits.getheader(img_sim_dir + '5deg_Y106_0_1.fits')
    hdr = dirimg_hdr

    ihdul = fits.HDUList()
    ext_sci = fits.ImageHDU(data=full_img, header=hdr, name='SCI')
    ihdul.append(ext_sci)
    # ext_err = fits.ImageHDU(data=np.sqrt(full_img), header=hdr, name='ERR')
    # ihdul.append(ext_err)
    # ext_dq = fits.ImageHDU(data=np.ones(full_img.shape),
    #                        header=hdr, name='DQ')
    # ihdul.append(ext_dq)
    ihdul.writeto(dirimage, overwrite=True)

    # ------- Now save segmap
    shdul = fits.HDUList()
    ext1 = fits.ImageHDU(data=segmap, header=hdr, name='SCI')
    shdul.append(ext1)
    shdul.writeto(segfile, overwrite=True)

    # -------- SEDLST
    with open(sedlst, 'w') as fh:
        fh.write("# 1: SEGMENTATION ID" + "\n")
        fh.write("# 2: SED FILE" + "\n")

        fh.write('1 7044.sed' + "\n")
        fh.write('2 5329.sed' + "\n")

    # -------- OBS
    obs_filt = 'hst_wfc3_f105w'

    with open(obslst, 'w') as fho:
        fho.write('# Image File name' + '\n')
        fho.write('# Observing band' + '\n')

        fho.write('\n' + dirimage + '  ' + obs_filt)

    # -------- WCS
    obs_ra = float(dirimg_hdr['CRVAL1'])
    obs_dec = float(dirimg_hdr['CRVAL2'])
    rollangles = [0.0]

    with open(wcslst, 'w') as fhw:

        hdr = ('# TELESCOPE = Roman' + '\n'
               '# INSTRUMENT = WFI' + '\n'
               '# DETECTOR = WFI' + '\n'
               '# GRISM = G150' + '\n'
               '# BLOCKING = ' + '\n')

        fhw.write(hdr + '\n')

        for r in range(len(rollangles)):
            obs_roll = rollangles[r]

            fhw.write('roman_grism_test_' + str(r+1))
            fhw.write('  ' + str(obs_ra))
            fhw.write('  ' + str(obs_dec))
            fhw.write('  ' + str(obs_roll))
            fhw.write('  G150' + '\n')

    # -------- Simulation
    sources = pylinear.source.SourceCollection(segfile, obslst,
                                               detindex=0, maglim=maglim)

    grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
    tabulate = pylinear.modules.Tabulate('pdt', ncpu=0)
    tabulate.run(grisms, sources, beam)

    simulate = pylinear.modules.Simulate(sedlst, gzip=False, ncpu=0)
    simulate.run(grisms, sources, beam)

    # -------- Add noise
    # Read in the noiseless image
    origname = 'roman_grism_test_1_flt.fits'
    orig_img = fits.open(origname)

    sci = orig_img['SCI'].data
    hdr = orig_img['SCI'].header
    size = sci.shape

    newsci = pylinearnoise(sci, size, exptime)

    # save
    fltname = origname.replace('.fits', '_noised.fits')
    orig_img['SCI'].data = newsci
    orig_img.writeto(fltname, overwrite=True)

    # -------- FLT
    with open(fltlst, 'w') as fh:
        hdr1 = "# Path to each flt image" + "\n"
        hdr2 = "# This has to be a simulated or " + \
               "observed dispersed image" + "\n"

        fh.write(hdr1)
        fh.write(hdr2)

        fh.write('\n' + fltname)

    # -------- Extraction
    grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
    tabulate = pylinear.modules.Tabulate('pdt', path=tablespath, ncpu=0)
    tabulate.run(grisms, sources, beam)

    extraction_parameters = grisms.get_default_extraction()

    extpar_fmt = 'Default parameters: range = {lamb0}, {lamb1} A,' + \
                 ' sampling = {dlamb} A'
    print("Default extraction parameters:")
    print(extpar_fmt.format(**extraction_parameters))

    # Set extraction params
    sources.update_extraction_parameters(**extraction_parameters)
    method = 'grid'  # golden, grid, or single
    extroot = 'roman_grism_test_'
    logdamp = [-6, -1, 0.1]

    pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp,
                                       method, extroot, tablespath,
                                       inverter='lsqr', ncpu=6,
                                       group=False)

    # -------- Read models
    model1 = np.genfromtxt('simulated_SEDs/1.sed', dtype=None,
                           names=['wav', 'flam'],
                           encoding='ascii', skip_header=3)
    model1_lam = model1['wav']
    model1_spec = model1['flam']

    model2 = np.genfromtxt('simulated_SEDs/2.sed', dtype=None,
                           names=['wav', 'flam'],
                           encoding='ascii', skip_header=3)
    model2_lam = model2['wav']
    model2_spec = model2['flam']

    # -------- Read and plot extracted spectrum
    x1d = fits.open(extroot + '_x1d.fits')

    spec1 = x1d[('SOURCE', 1)].data
    spec2 = x1d[('SOURCE', 2)].data
    pylinear_flam_scale_fac = 1e-17

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
    ax1.set_title('7044')
    ax2.set_title('5329')

    ax1.plot(spec1['wavelength'], spec1['flam'] * pylinear_flam_scale_fac,
             color='k', lw=1)
    ax1.plot(model1_lam, model1_spec, ls='--', color='skyblue', lw=0.5)

    ax2.plot(spec2['wavelength'], spec2['flam'] * pylinear_flam_scale_fac,
             color='k', lw=1)
    ax2.plot(model2_lam, model2_spec, ls='--', color='skyblue', lw=0.5)

    fig.savefig('roman_grism_test_x1d.png', dpi=200, bbox_inches='tight')

    sys.exit(0)
