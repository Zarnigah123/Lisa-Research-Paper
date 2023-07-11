#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:31:16 2022

@author: nazeela
"""

import h5py as h5
import numpy as np
import astropy.units as u
import getopt
import sys

from compas_processing import get_COMPAS_vars, mask_COMPAS_data
import legwork as lw

from galaxy import simulate_mw


SNR_CUTOFF = 7
MW_SIZE = 1_000_000


def usage():
    ansi_codes = {
        "green": '\033[92m',
        "red": '\u001b[31m',
        "reset": '\033[0m'
    }
    print()
    print("{}usage:{} python simulation_DCO_detections.py {}[options]{}".format(ansi_codes["green"],
                                                                                ansi_codes["reset"],
                                                                                ansi_codes["red"],
                                                                                ansi_codes["reset"]))
    print()
    print("{}options:{}".format(ansi_codes["red"], ansi_codes["reset"]))
    print("      Option names      :                                     Description                                     :       Default value      ")
    print("------------------------:-------------------------------------------------------------------------------------:--------------------------")
    print("  -h, --help            : print usage instructions                                                            : -")
    print("  -i, --input           : path to COMPAS h5 input file                                                        : 'COMPASOutput.h5'")
    print("  -o, --output          : path to output h5 file                                                              : 'COMPASOutput_testing.h5'")
    print("  -n, --loops           : number of simulations to run                                                        : 10")
    print("  -t, --binary-type     : which binary type to simulation ['BHBH', 'BHNS', 'NSNS']                            : 'BHNS'")
    print("  -fig_, --opt-flag        : whether to use the optimistic CE scenario                                           : -")
    print("  -s, --simple-mw       : whether to use the simple MW model                                                  : -")
    print("  -e, --extended-mission: whether to assume an extended (10 yr) LISA mission                                  : -")
    print("  -b, --case-bb-survive : whether to allow case BB systems to survive CE whilst using pessimistic CE scenario : -")
    print()

#####################################################################


def main():
    # get command line arguments and exit if error
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hi:o:n:t:fseb", ["help",
                                                                "input=",
                                                                "output=",
                                                                "loops=",
                                                                "binary-type=",
                                                                "opt-flag",
                                                                "simple-mw",
                                                                "extended-mission",
                                                                "case-bb-survive"])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    # set default values
    input_filepath = '/home/nazeela/NewCodesAliMohsin/data/h5out.h5'
    loops = 100
    binary_type = "BHBH"
    pessimistic = False
    use_simple_mw = False
    t_obs = 4 * u.yr
    allow_caseBB_survive = False

    # change defaults based on input
    for option, value in opts:
        if option in ("-h", "--help"):
            usage()
            return
        if option in ("-i", "--input"):
            input_filepath = value
        if option in ("-o", "--output"):
            output_filepath = value
        if option in ("-n" or "--loops"):
            loops = int(value)
        if option in ("-t", "--binary-type"):
            binary_type = value
        if option in ("-fig_", "--opt-flag"):
            pessimistic = False
        if option in ("-s", "--simple-mw"):
            use_simple_mw = True
        if option in ("-e", "--extended-mission"):
            t_obs = 10 * u.yr
        if option in ("-b", "--case-bb-survive"):
            allow_caseBB_survive = True

    if not pessimistic:
        # no need to do case BB stuff if already optimistic
        allow_caseBB_survive = False
    else:
        # force optimistic if letting case BB to survive so we can mask them away again
        pessimistic = False if allow_caseBB_survive else True

    # open COMPAS file
    with h5.File(input_filepath, "r") as COMPAS_file:
        # mask only required DCOs

        # _h5 = pd.DataFrame(h5.File('metal.h5', 'r')['simulation'][...].squeeze())
        _h5 = COMPAS_file['BSE_System_Parameters.csv']
        metali=np.array(_h5['Metallicity@ZAMS(1)'][()])
        # mseed=np.array(_h5['SEED'][()])
        # get all relevant variables

        dco_mask = mask_COMPAS_data(COMPAS_file,binary_type, (True, False, False))

        dco_mask = np.where(dco_mask == 1)[0]
        # dco_mask=[True if i==1 else False for i in dco_mask]
        # print(len(dco_mask))

        [compas_m_1, compas_m_2,\
            compas_a_DCO,\
            compas_e_DCO, compas_t_evol,\
            compas_seeds], compas_Z = get_COMPAS_vars(COMPAS_file,
                                           "BSE_Double_Compact_Objects.csv",
                                           ["Mass(1)", "Mass(2)",
                                            "SemiMajorAxis@DCO",
                                            "Eccentricity@DCO",
                                            "Time", "SEED"],mask=dco_mask,
                                           metallicity=metali)

        # add units
        compas_m_1, compas_m_2 = compas_m_1 * u.Msun, compas_m_2 * u.Msun
        compas_a_DCO *= u.AU
        compas_t_evol *= u.Myr


    # find unique metallicities
    #compas_Z_unique = compas_Z

    # allow case BB systems to survive the CE even when pessimistic
    # if allow_caseBB_survive:

    #     # start a mask of which binaries to throw away
    #     exclude = np.repeat(False, compas_seeds.shape)

    #     # do it separately by metallicity to ensure seeds are unique
    #     for Z in compas_Z_unique:
    #         # WARNING: I have hardcoded the use of the unstable case BB file here
    #         # this will produce (possibly silent) errors if other models are use
    #         ce_path = "/n/holystore01/LABS/berger_lab/Lab/fbroekgaarden/DATA/all_dco_legacy_CEbug_fix/"
    #         ce_path += "unstableCaseBB/Z_{}/STROOPWAFELcombined/COMPASOutput.h5".format(Z)

    #         # grab the CE file data for this metallicity bin
    #         with h5.File(ce_path, "r") as ce_file:
    #             ce_seeds, ce_st1, ce_st2 = get_COMPAS_vars(ce_file, "commonEnvelopes",
    #                                                        ["randomSeed", "stellarType1", "stellarType2"])

    #         # make a mask that just excludes HG but not HeHG
    #         ce_with_HG = np.logical_or(ce_st1 == 2, ce_st2 == 2)

    #         # get the corresponding seeds (unique in case there were multiple CE events)
    #         seeds_to_delete = np.unique(ce_seeds[ce_with_HG])

    #         # add to the mask
    #         dco_matching_Z = compas_Z == Z
    #         exclude[dco_matching_Z] = np.isin(compas_seeds[dco_matching_Z], seeds_to_delete)

    #     # mask the main arrays of all HG CEEs
    #     compas_m_1, compas_m_2, compas_Z, compas_a_DCO, compas_e_DCO,\
    #         compas_t_evol, compas_weights, compas_seeds = compas_m_1[exclude], compas_m_2[exclude],\
    #         compas_Z[exclude], compas_a_DCO[exclude], compas_e_DCO[exclude], compas_t_evol[exclude],\
    #         compas_weights[exclude], compas_seeds[exclude]

    # work out metallicity bins
    compas_Z_unique = np.unique(compas_Z)
    inner_bins = np.sort(np.array([compas_Z_unique[i] + (compas_Z_unique[i+1] - compas_Z_unique[i]) / 2
                           for i in range(len(compas_Z_unique) - 1)]))

    Z_bins = np.concatenate(([compas_Z_unique[0]], inner_bins,
                             [compas_Z_unique[-1]]))

    # create a random number generator
    rng = np.random.default_rng()

    # prep the temporary variable for parameters
    MAX_HIGH = 500
    dt = np.dtype(float)
    dtype = [("m_1", dt), ("m_2", dt), ("a_DCO", dt), ("e_DCO", dt),
             ("a_LISA", dt), ("e_LISA", dt), ("t_evol", dt), ("t_merge", dt),
             ("tau", dt), ("Z", dt), ("component", np.dtype(int)),
             ("snr", dt), ("weight", dt), ("seed", dt),
             ("dist", dt), ("R", dt), ("z", dt), ("theta", dt), ('gal', dt)]
    to_file = np.zeros(shape=(loops * MAX_HIGH,), dtype=dtype)

    n_detect_list = np.zeros(loops)
    total_MW_weight = np.zeros(loops)
    tot_detect = 0
    for milky_way in range(loops):
        output_filepath = '/home/nazeela/NewCodesAliMohsin/data/COMPAS_Output.h5'

        mw_comp = simulate_mw(MW_SIZE, ret_pos=True)

        tau, dist, Z_unbinned, pos, component = mw_comp

        component[component == "low_alpha_disc"] = 0
        component[component == "high_alpha_disc"] = 1
        component[component == "bulge"] = 2

        R, z, theta = pos

        # np.save(f'galaxy_{milky_way+1}.npy', mw_comp, allow_pickle=True)


        # work out COMPAS limits (and limit to Z=0.022)
        min_Z_compas = np.min(compas_Z_unique)

        max_Z_compas = np.max(compas_Z_unique[compas_Z_unique <= 0.03])
        # max_Z_compas = 0.03

        # change metallicities above COMPAS limits to between solar and upper
        too_big = Z_unbinned > max_Z_compas

        Z_unbinned[too_big] = 10**(np.random.uniform(np.log10(np.min([0.01416, max_Z_compas])),
                                                     np.log10(max_Z_compas),
                                                     len(Z_unbinned[too_big])))

        # change metallicities below COMPAS limits to lower limit
        too_small = Z_unbinned < min_Z_compas

        # print(Z_unbinned[too_small])
        Z_unbinned[too_small] = min_Z_compas

        # sort by metallicity so everything matches up well
        Z_order = np.argsort(Z_unbinned)
        tau, dist, Z_unbinned, R, z, theta, component = tau[Z_order], dist[Z_order], Z_unbinned[Z_order],\
            R[Z_order], z[Z_order], theta[Z_order], component[Z_order]

        # bin the metallicities using Floor's bins
        h, _ = np.histogram(Z_unbinned, bins=Z_bins)

        # draw binaries for each metallicity bin, store indices
        binaries = np.zeros(MW_SIZE).astype(int)
        indices = np.arange(len(compas_m_1)).astype(int)
        total = 0

        # print('starting binary placement')

        for i in range(len(h)):
            if h[i] > 0:
                same_Z = compas_Z == compas_Z_unique[i]
                binaries[total:total + h[i]] = rng.choice(indices[same_Z], h[i], replace=True)
                total += h[i]


        # TODO: remove this eventually
        if total != MW_SIZE:
            print(compas_Z_unique)
            print(Z_bins)
            print(np.sum(h), h)
            print(min_Z_compas, max_Z_compas)
            exit("PANIC: something funky is happening with the Z bins")

        # mask parameters for binaries
        m_1=compas_m_1[binaries]
        m_2=compas_m_2[binaries]
        a_DCO=compas_a_DCO[binaries]
        e_DCO=compas_e_DCO[binaries]
        t_evol=compas_t_evol[binaries]
        Z=compas_Z[binaries]
        seed = compas_seeds[binaries]
        w = np.ones(len(m_1))
        gal = np.zeros(len(m_1)) + (milky_way + 1)

        # store the total weight of full population (for normalisation)
        #total_MW_weight[milky_way] = np.sum(w)

        # work out which binaries are still inspiralling
        t_merge = lw.evol.get_t_merge_ecc(ecc_i=e_DCO, a_i=a_DCO, m_1=m_1, m_2=m_2)
        insp = t_merge > (tau - t_evol)

        # trim out the merged binaries
        m_1, m_2, a_DCO, e_DCO, t_evol, t_merge, tau, dist, Z, R, z, theta, component, w, seed, gal = m_1[insp],\
            m_2[insp], a_DCO[insp], e_DCO[insp], t_evol[insp], t_merge[insp], tau[insp], dist[insp], Z[insp],\
            R[insp], z[insp], theta[insp], component[insp], w[insp], seed[insp], gal[insp]

        # evolve binaries to LISA
        e_LISA, a_LISA, f_orb_LISA = lw.evol.evol_ecc(ecc_i=e_DCO, a_i=a_DCO, m_1=m_1, m_2=m_2, t_evol=tau - t_evol, output_vars=["ecc", "a", "f_orb"], n_proc=6)
        # we only care about the final state
        e_LISA = e_LISA[:, -1]
        a_LISA = a_LISA[:, -1]
        f_orb_LISA = f_orb_LISA[:, -1]

        sources = lw.source.Source(m_1=m_1, m_2=m_2, ecc=e_LISA, dist=dist, f_orb=f_orb_LISA, sc_params={"t_obs": t_obs})
        snr = sources.get_snr(t_obs=4*u.yr)

        detectable = snr > SNR_CUTOFF
        n_detect = len(snr[detectable])
        print(fig_'n_detect={n_detect}')
        print(binary_type,n_detect,'Properties')
        n_detect_list[milky_way] = n_detect

        # store parameters in temporary variable
        to_file["m_1"][tot_detect:tot_detect + n_detect] = m_1[detectable]
        to_file["m_2"][tot_detect:tot_detect + n_detect] = m_2[detectable]
        to_file["a_DCO"][tot_detect:tot_detect + n_detect] = a_DCO[detectable]
        to_file["e_DCO"][tot_detect:tot_detect + n_detect] = e_DCO[detectable]
        to_file["a_LISA"][tot_detect:tot_detect + n_detect] = a_LISA[detectable]
        to_file["e_LISA"][tot_detect:tot_detect + n_detect] = e_LISA[detectable]
        to_file["t_evol"][tot_detect:tot_detect + n_detect] = t_evol[detectable]
        to_file["t_merge"][tot_detect:tot_detect + n_detect] = t_merge[detectable]
        to_file["tau"][tot_detect:tot_detect + n_detect] = tau[detectable]
        to_file["component"][tot_detect:tot_detect + n_detect] = component[detectable]
        to_file["Z"][tot_detect:tot_detect + n_detect] = Z[detectable]
        to_file["snr"][tot_detect:tot_detect + n_detect] = snr[detectable]
        to_file["weight"][tot_detect:tot_detect + n_detect] = w[detectable]
        to_file["seed"][tot_detect:tot_detect + n_detect] = seed[detectable]
        to_file["dist"][tot_detect:tot_detect + n_detect] = dist[detectable]
        to_file["R"][tot_detect:tot_detect + n_detect] = R[detectable]
        to_file["z"][tot_detect:tot_detect + n_detect] = z[detectable]
        to_file["theta"][tot_detect:tot_detect + n_detect] = theta[detectable]
        to_file['gal'][tot_detect:tot_detect + n_detect] = gal[detectable]

        tot_detect += n_detect

    to_file = to_file[:tot_detect]

    # store all parameters in h5 file
    with h5.File(output_filepath, "w") as file:
        file.create_dataset("simulation", (tot_detect,), dtype=dtype)
        file["simulation"][...] = to_file
        file["simulation"].attrs["n_detect"] = n_detect_list
        file["simulation"].attrs["total_MW_weight"] = total_MW_weight


if __name__ == "__main__":
    main()