"""
Created on Thu Feb 10 11:31:16 2022

@author: nazeela
"""
import os

import astropy.units as u
import h5py as h5
import legwork as lw
import numpy as np
import pandas as pd
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table

from compas_processing import get_COMPAS_vars, mask_COMPAS_data
from galaxy import simulate_mw

np.random.seed(235)

input_filepath = "/media/astrophysicsandpython/DATA_DRIVE0/h5out.h5"
loops = 100
binary_type = ['NSNS', 'BHNS', 'NSBH']
t_obs = 4 * u.yr
MW_SIZE = 1_000_000
SNR_CUTOFF = 7


#####################################################################

def chunks(lst, n):
    # taken from https://stackoverflow.com/a/312464/3212945
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def make_h5(file_name, dataframe, order=None, keys=None, folder_name="simulation"):
    if order is None:
        new_dataframe = dataframe
        order = keys
    else:
        new_dataframe = dataframe.reindex(
            columns=order)  # taken from https://stackoverflow.com/a/47467999/3212945

    h5_ = h5.File(file_name, "w")
    write_table_hdf5(Table(data=np.stack(np.array(new_dataframe.T), axis=1), names=order), h5_,
                     folder_name)
    h5_.close()


def make_detectable_dataset(input_filepath, loops, binary_type, t_obs, MW_SIZE=1_000_000,
                            SNR_CUTOFF=7):
    with h5.File(input_filepath, "r") as COMPAS_file:
        _h5 = COMPAS_file["BSE_System_Parameters.csv"]
        metallicity_ = np.array(_h5["Metallicity@ZAMS(1)"][()])

        dco_mask = mask_COMPAS_data(COMPAS_file, binary_type, (True, False, False))

        dco_mask = np.where(dco_mask == 1)[0]

        # sys_par = COMPAS_file["BSE_System_Parameters.csv"]
        zams_m_1 = _h5["Mass@ZAMS(1)"][()]
        zams_m_2 = _h5["Mass@ZAMS(2)"][()]
        zams_a = _h5["SemiMajorAxis@ZAMS"][()]
        zams_e = _h5["Eccentricity@ZAMS"][()]
        zams_seeds = _h5["SEED"][()]

        _, compas_Z = get_COMPAS_vars(COMPAS_file, "BSE_Double_Compact_Objects.csv",
                                      ["Mass(1)", "Mass(2)", "SemiMajorAxis@DCO",
                                       "Eccentricity@DCO",
                                       "Time", "SEED"],
                                      mask=dco_mask,
                                      metallicity=metallicity_)

        compas_m_1, compas_m_2, compas_a_DCO, compas_e_DCO, compas_t_evol, compas_seeds = _

        # add units
        compas_m_1, compas_m_2 = compas_m_1 * u.Msun, compas_m_2 * u.Msun
        compas_a_DCO *= u.AU
        compas_t_evol *= u.Myr

    index_1 = np.in1d(zams_seeds, compas_seeds)

    zams_m_1 = zams_m_1[index_1]
    zams_m_2 = zams_m_2[index_1]
    zams_a = zams_a[index_1]
    zams_e = zams_e[index_1]
    zams_seeds = zams_seeds[index_1]

    temp1_ = pd.DataFrame([zams_m_1, zams_m_2, zams_a, zams_e, zams_seeds]).T
    temp1_.columns = ["Mass@ZAMS(1)", "Mass@ZAMS(2)", "SemiMajorAxis@ZAMS", "Eccentricity@ZAMS",
                      "SEEDS@ZAMS"]

    zams_m1, zams_m2, zams_a, zams_e, zams_seeds = temp1_.to_numpy().T

    # work out metallicity bins
    compas_Z_unq = np.unique(compas_Z)
    inner_bins = np.sort(np.array([compas_Z_unq[i] + (compas_Z_unq[i + 1] - compas_Z_unq[i]) / 2
                                   for i in range(len(compas_Z_unq) - 1)]))

    Z_bins = np.concatenate(([compas_Z_unq[0]], inner_bins, [compas_Z_unq[-1]]))

    # create a random number generator
    rng = np.random.default_rng()

    # prep the temporary variable for parameters
    MAX_HIGH = 500
    dt = np.dtype(float)

    n_detect_list = np.zeros(loops)
    total_MW_weight = np.zeros(loops)
    tot_detect = 0

    for milky_way in range(26, loops):
        print(f"number {milky_way + 1}\n")

        output_filepath = f"./data/COMPAS_Output_{binary_type}.h5"

        tau, dist, Z_unbinned, pos, component = simulate_mw(MW_SIZE, ret_pos=True)

        component[component == "low_alpha_disc"] = 0
        component[component == "high_alpha_disc"] = 1
        component[component == "bulge"] = 2

        R, z, theta = pos

        # work out COMPAS limits (and limit to Z=0.022)
        min_Z_compas = np.min(compas_Z_unq)

        max_Z_compas = np.max(compas_Z_unq[compas_Z_unq <= 0.03])
        # max_Z_compas = 0.03

        # change metallicities above COMPAS limits to between solar and upper
        too_big = Z_unbinned > max_Z_compas

        Z_unbinned[too_big] = 10**(np.random.uniform(np.log10(0.01416),
                                                     np.log10(max_Z_compas),
                                                     len(Z_unbinned[too_big])))

        # change metallicities below COMPAS limits to lower limit
        too_small = Z_unbinned < min_Z_compas

        # print(Z_unbinned[too_small])
        Z_unbinned[too_small] = 10**(np.random.uniform(np.log10(min_Z_compas),
                                                       np.log10(0.01416),
                                                       len(Z_unbinned[too_small])))

        # sort by metallicity so everything matches up well
        Z_order = np.argsort(Z_unbinned)
        tau, dist, Z_unbinned, R, z, theta, component = tau[Z_order], dist[Z_order], Z_unbinned[
            Z_order], R[Z_order], z[Z_order], theta[Z_order], component[Z_order]

        # bin the metallicities using Floor's bins
        h, _ = np.histogram(Z_unbinned, bins=Z_bins)

        # draw binaries for each metallicity bin, store indices
        binaries = np.zeros(MW_SIZE).astype(int)
        indices = np.arange(len(compas_m_1)).astype(int)
        total = 0

        # print("starting binary placement")

        for i, v in enumerate(h):
            if h[i] > 0:
                same_Z = compas_Z == compas_Z_unq[i]
                binaries[total:total + h[i]] = rng.choice(indices[same_Z], h[i], replace=True)
                total += h[i]

        if total != MW_SIZE:
            print(compas_Z_unq)
            print(Z_bins)
            print(np.sum(h), h)
            print(min_Z_compas, max_Z_compas)
            exit("PANIC: something funky is happening with the Z bins")

        # mask parameters for binaries
        sys_m1 = zams_m1[binaries]
        sys_m2 = zams_m2[binaries]
        sys_a = zams_a[binaries]
        sys_e = zams_e[binaries]
        sys_seeds = zams_seeds[binaries]

        dco_m1 = compas_m_1[binaries]
        dco_m2 = compas_m_2[binaries]
        dco_a = compas_a_DCO[binaries]
        dco_e = compas_e_DCO[binaries]
        t_evol = compas_t_evol[binaries]
        Z = compas_Z[binaries]
        seed = compas_seeds[binaries]
        w = np.ones(len(dco_m1))

        print("starting t_merge")

        # work out which binaries are still inspiralling
        e_dco_ = chunks(dco_e, 1000)
        a_dco_ = chunks(dco_a, 1000)
        m1 = chunks(dco_m1, 1000)
        m2 = chunks(dco_m2, 1000)

        merge = [lw.evol.get_t_merge_ecc(ecc_i=i, a_i=j, m_1=k, m_2=l).value for i, j, k, l in
                 zip(e_dco_, a_dco_, m1, m2)]

        print("t_merge ended")

        t_merge = np.array([element for innerList in merge for element in innerList])
        t_merge = t_merge * u.Gyr
        inspiral = t_merge > (tau - t_evol)

        # trim out the merged binaries
        dco_m1, dco_m2 = dco_m1[inspiral], dco_m2[inspiral]
        dco_a, dco_e = dco_a[inspiral], dco_e[inspiral]
        t_evol, t_merge = t_evol[inspiral], t_merge[inspiral]
        tau, dist, Z, component = tau[inspiral], dist[inspiral], Z[inspiral], component[inspiral]
        R, z, theta = R[inspiral], z[inspiral], theta[inspiral]
        w, seed = w[inspiral], seed[inspiral]

        sys_m1, sys_m2 = sys_m1[inspiral], sys_m2[inspiral]
        sys_a, sys_e, sys_seeds = sys_a[inspiral], sys_e[inspiral], sys_seeds[inspiral]

        # evolve binaries to LISA
        e_LISA, a_LISA, f_orb_LISA = lw.evol.evol_ecc(ecc_i=dco_e, a_i=dco_a, m_1=dco_m1,
                                                      m_2=dco_m2,
                                                      t_evol=tau - t_evol,
                                                      output_vars=["ecc", "a", "f_orb"], n_proc=3)
        # we only care about the final state
        e_LISA = e_LISA[:, -1]
        a_LISA = a_LISA[:, -1]
        f_orb_LISA = f_orb_LISA[:, -1]

        sources = lw.source.Source(m_1=dco_m1, m_2=dco_m2, ecc=e_LISA, dist=dist, f_orb=f_orb_LISA,
                                   sc_params={"t_obs": t_obs})
        snr = sources.get_snr(t_obs=4 * u.yr)
        harmonics = sources.max_snr_harmonic

        detectable = snr > SNR_CUTOFF
        n_detect = len(snr[detectable])
        print(f"n_detect={n_detect}")
        print(binary_type, n_detect, "Properties")
        n_detect_list[milky_way] = n_detect

        dtype = [("m1_zams", dt), ("m1_dco", dt), ("m2_zams", dt), ("m2_dco", dt), ("m_chrip", dt),
                 ("a_zams", dt), ("a_dco", dt), ("a_lisa", dt), ("f_orb", dt),
                 ("e_zams", dt), ("e_dco", dt), ("e_lisa", dt),
                 ("t_evol", dt), ("t_merge", dt), ("lookback_time", dt), ("Z", dt),
                 ("component", np.dtype(int)), ("SNR", dt), ("SNR_harmonics", dt),
                 ("weight", np.dtype(int)),
                 ("distance", dt), ("R", dt), ("z", dt), ("theta", dt), ("seed", np.dtype(int))]

        to_file = np.zeros(n_detect, dtype=dtype)

        # store parameters in temporary variable
        to_file["m1_zams"] = sys_m1[detectable]
        to_file["m1_dco"] = dco_m1[detectable]
        to_file["m2_zams"] = sys_m2[detectable]
        to_file["m2_dco"] = dco_m2[detectable]
        to_file["m_chrip"] = lw.utils.chirp_mass(dco_m1[detectable], dco_m2[detectable])
        to_file["a_zams"] = sys_a[detectable]
        to_file["a_dco"] = dco_a[detectable]
        to_file["a_lisa"] = a_LISA[detectable]
        to_file["f_orb"] = f_orb_LISA[detectable]
        to_file["e_zams"] = sys_e[detectable]
        to_file["e_dco"] = dco_e[detectable]
        to_file["e_lisa"] = e_LISA[detectable]
        to_file["t_evol"] = t_evol[detectable].to('Myr')
        to_file["t_merge"] = t_merge[detectable].to('Myr')
        to_file["lookback_time"] = tau[detectable].to('Myr')
        to_file["Z"] = Z[detectable]
        to_file["component"] = component[detectable]
        to_file["SNR"] = snr[detectable]
        to_file["SNR_harmonics"] = harmonics[detectable]
        to_file["distance"] = dist[detectable]
        to_file["R"] = R[detectable]
        to_file["z"] = z[detectable]
        to_file["theta"] = theta[detectable]
        to_file["weight"] = w[detectable]

        to_file["seed"] = sys_seeds[detectable]

        df = pd.DataFrame(to_file)
        df = df.loc[~(df == 0).all(axis=1)]

        cwd = os.getcwd()
        dir_ = f'./new{binary_type.lower()}'

        os.makedirs(f'{dir_}', exist_ok=True)

        os.chdir(f'{dir_}')

        make_h5(f"{binary_type}__GalaxyNum_{milky_way + 1:03}.h5", df,
                keys=["m1_zams", "m1_dco", "m2_zams", "m2_dco", "m_chirp",
                      "a_zams", "a_dco", "a_lisa",
                      "f_orb",
                      "e_zams", "e_dco", "e_lisa",
                      "t_evol", "t_merge", "lookback_time",
                      "Z", "component",
                      "SNR", "SNR_harmonics",
                      "distance", "R", "z", "theta",
                      "weight", "seed"])
        os.chdir(cwd)


make_detectable_dataset(input_filepath, loops, 'BHNS', t_obs, MW_SIZE, SNR_CUTOFF)
