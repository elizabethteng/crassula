#!/usr/bin/env python

import numpy as np
import os
import pickle
from time import time
import pdspy.modeling as modeling
import pdspy.dust as dust
import numpy as np

ranges = [[3000.,5000.], [-1,3.],[-8.,-2.], [0.,3.],[0.01,0.5],[-1.,2.5],[0.0,2.0],\
        [0.5,2.0],[-8.,-2.],[2.5,4.], [0.,1.], [0.5,1.5],[0.,5.],[2.5,4.5],[0.,90.]]
steps=[]
bases=[]
for i in range(len(ranges)):
    steps.append(np.linspace(ranges[i][0],ranges[i][1],11))
    bases.append(steps[i][5])
pars=bases

def run_yso_model( Tstar=None, logL_star=None, \
        logM_disk=None, logR_disk=None, h_0=None, logR_in=None, gamma=None, \
        beta=None, logM_env=None, logR_env=None, f_cav=None, ksi=None, \
        loga_max=None, p=None, incl=None):
    
    params = [Tstar,logL_star,logM_disk,logR_disk,h_0,logR_in,\
          gamma,beta,logM_env,logR_env,f_cav,ksi,loga_max,p,incl]

    param_names = ['Tstar','logLstar','logMdisk','logRdisk','h0','logRin',\
          'gamma','beta','logMenv','logRenv','fcav','ksi','logamax','p','incl']
    # Set up the dust properties.

    dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/diana_wice.hdf5")
    ddust = dust_gen(10.**loga_max / 1e4, p)
    env_dust_gen = dust.DustGenerator(dust.__path__[0]+\
            "/data/diana_wice.hdf5")
    edust = env_dust_gen(1.0e-4, 3.5)

    # Calculate alpha correctly.

    alpha = gamma + beta

    # Fix the scale height of the disk.

    h_0 *= (10.**logR_disk)**beta

    # Set up the model.

    model = modeling.YSOModel()
    model.add_star(luminosity=10.**logL_star, temperature=Tstar)
    model.set_spherical_grid(10.**logR_in, 10.**logR_env, 100, 101, 2, \
            code="radmc3d")
    model.add_pringle_disk(mass=10.**logM_disk, rmin=10.**logR_in, \
            rmax=10.**logR_disk, plrho=alpha, h0=h_0, plh=beta, dust=ddust)
    model.add_ulrich_envelope(mass=10.**logM_env, rmin=10.**logR_in, \
            rmax=10.**logR_env, cavpl=ksi, cavrfact=f_cav, dust=edust)
    model.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)
 

    print("finished setting up model, now running thermal simulation")

    # Run the thermal simulation
    model.run_thermal(code="radmc3d", nphot=1e6, \
            modified_random_walk=True, verbose=False, setthreads=17, \
            timelimit=10800)
    
    print("finished running thermal simulation, now running SED")
    t2=time()
    
    # Run the SED.

    model.set_camera_wavelength(np.logspace(-1.,4.,500))
    model.run_sed(name="SED", nphot=1e5, loadlambda=True, incl=incl, \
            pa=0., dpc=140., code="radmc3d", camera_scatsrc_allfreq=True, \
            verbose=False, setthreads=17)
    
    filename=""
    for i in range(len(param_names)):
        filename+=param_names[i]+"_"
        filename+=str(params[i])+"_"
    filename=filename[:-1]
    filename+=".hdf5"
    print("finished running "+filename[0:40]+"... in %0.3fs" % (time() - t2))
    
    # Write out the file.
    model.write_yso("../../grid/transform/"+filename)

gammafill=[1.66,1.733, 1.84,1.88,1.92,1.96]
logmenvfill=[-3.6,-3.4,-3.0,-2.8,-2.48,-2.36,-2.24,-2.12]


#for i in range(2,2,n(gammafill)):
#    t0=time()
#    run_yso_model(Tstar=pars[0], logL_star=pars[1], \
#            logM_disk=pars[2], logR_disk=pars[3], h_0=pars[4], logR_in=pars[5], gamma=gammafill[i],\
#            beta=pars[7], logM_env=pars[8], logR_env=pars[9], f_cav=pars[10], ksi=pars[11], \
#            loga_max=pars[12], p=pars[13], incl=pars[14])
#    print("finished running SED #"+str(i)+" in %0.3fs" % (time() - t0))

for i in range(4,len(logmenvfill)):
    t0=time()
    run_yso_model(Tstar=pars[0], logL_star=pars[1], \
            logM_disk=pars[2], logR_disk=pars[3], h_0=pars[4], logR_in=pars[5], gamma=pars[6],\
            beta=pars[7], logM_env=logmenvfill[i], logR_env=pars[9], f_cav=pars[10], ksi=pars[11], \
            loga_max=pars[12], p=pars[13], incl=pars[14])

    print("finished running SED #"+str(i)+" in %0.3fs" % (time() - t0))


