import itertools

from matplotlib import pyplot as plt
from matplotlib.pyplot import rc

from utilities import get_mean_std, H5Files

rc('font', **{'size': 14})

h5_files = H5Files('./../')

bhbh_df = h5_files.get_bhbh_file
nsns_df = h5_files.get_bhns_file
bhns_df = h5_files.get_bhns_file
nsbh_df = h5_files.get_nsbh_file

emz_bhbh, esz_bhbh = get_mean_std(bhbh_df, 'e_zams')
emz_nsns, esz_nsns = get_mean_std(nsns_df, 'e_zams')
emz_bhns, esz_bhns = get_mean_std(bhns_df, 'e_zams')
emz_nsbh, esz_nsbh = get_mean_std(nsbh_df, 'e_zams')

emd_bhbh, esd_bhbh = get_mean_std(bhbh_df, 'e_dco')
emd_nsns, esd_nsns = get_mean_std(nsns_df, 'e_dco')
emd_bhns, esd_bhns = get_mean_std(bhns_df, 'e_dco')
emd_nsbh, esd_nsbh = get_mean_std(nsbh_df, 'e_dco')

eml_bhbh, esl_bhbh = get_mean_std(bhbh_df, 'e_lisa')
eml_nsns, esl_nsns = get_mean_std(nsns_df, 'e_lisa')
eml_bhns, esl_bhns = get_mean_std(bhns_df, 'e_lisa')
eml_nsbh, esl_nsbh = get_mean_std(nsbh_df, 'e_lisa')

x = range(1, 101)

f, ax = plt.subplots(4, 1, sharex='all', figsize=(12, 14), sharey='col')


def plot_ecc(x_val, y_val, y_err, axes_, label, binary_type):
    color = ['r', 'b', 'g']

    [axes_.plot(x_val, i, marker='*', label=j, color=k, ls='')
     for i, j, k in zip(y_val, label, color)]

    [axes_.fill_between(x_val, i - j, i + j, alpha=0.1, color=k)
     for i, j, k in zip(y_val, y_err, color)]

    temp_ = [i for i in zip(*y_val)]
    sorted_ = [list(zip(sorted(i, reverse=True))) for i in temp_]
    sorted_y = [list(itertools.chain.from_iterable(i)) for i in sorted_]

    l1 = [i[0] for i in sorted_y]
    l2 = [i[1] for i in sorted_y]
    l3 = [i[2] for i in sorted_y]

    axes_.plot([x_val] * len(y_val), [l1, l2, l3], ls=':', color='y', zorder=-1)

    axes_.set_ylabel(binary_type)


def return_raw_label(string):
    return r'$e_{' + f'{string}' + r'}$'


type_list = ['ZAMS', 'DCO', 'LISA']
type_list2 = ['', '', '']

plot_ecc(x, [emz_bhbh, emd_bhbh, eml_bhbh], [esz_bhbh, esd_bhbh, esl_bhbh], ax[0], type_list,
         return_raw_label('BHBH'))
plot_ecc(x, [emz_nsns, emd_nsns, eml_nsns], [esz_nsns, esd_nsns, esl_nsns], ax[1], type_list2,
         return_raw_label('NSNS'))
plot_ecc(x, [emz_nsbh, emd_nsbh, eml_nsbh], [esz_nsbh, esd_nsbh, esl_nsbh], ax[2], type_list2,
         return_raw_label('NSBH'))
plot_ecc(x, [emz_bhns, emd_bhns, eml_bhns], [esz_bhns, esd_bhns, esl_bhns], ax[3], type_list2,
         return_raw_label('BHNS'))

f.legend(loc='upper center', ncols=3)
plt.xlabel('Number of galaxies')
plt.tight_layout()
f.subplots_adjust(top=0.96)
plt.savefig('eccentricity_comparison.pdf')
plt.close()
