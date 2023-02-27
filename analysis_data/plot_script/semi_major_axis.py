import itertools

from matplotlib import pyplot as plt, rc

from utilities import get_mean_std, H5Files

rc('font', **{'size': 14})

h5_files = H5Files('./../')

bhbh_df = h5_files.get_bhbh_file
nsns_df = h5_files.get_bhns_file
bhns_df = h5_files.get_bhns_file
nsbh_df = h5_files.get_nsbh_file

amz_bhbh, asz_bhbh = get_mean_std(bhbh_df, 'a_zams')
amz_nsns, asz_nsns = get_mean_std(nsns_df, 'a_zams')
amz_bhns, asz_bhns = get_mean_std(bhns_df, 'a_zams')
amz_nsbh, asz_nsbh = get_mean_std(nsbh_df, 'a_zams')

amd_bhbh, asd_bhbh = get_mean_std(bhbh_df, 'a_dco')
amd_nsns, asd_nsns = get_mean_std(nsns_df, 'a_dco')
amd_bhns, asd_bhns = get_mean_std(bhns_df, 'a_dco')
amd_nsbh, asd_nsbh = get_mean_std(nsbh_df, 'a_dco')

aml_bhbh, asl_bhbh = get_mean_std(bhbh_df, 'a_lisa')
aml_nsns, asl_nsns = get_mean_std(nsns_df, 'a_lisa')
aml_bhns, asl_bhns = get_mean_std(bhns_df, 'a_lisa')
aml_nsbh, asl_nsbh = get_mean_std(nsbh_df, 'a_lisa')

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
    return r'$a_{' + f'{string}' + r'}$'


type_list = ['ZAMS', 'DCO', 'LISA']
type_list2 = ['', '', '']

plot_ecc(x, [amz_bhbh, amd_bhbh, aml_bhbh], [asz_bhbh, asd_bhbh, asl_bhbh], ax[0], type_list,
         return_raw_label('BHBH'))
plot_ecc(x, [amz_nsns, amd_nsns, aml_nsns], [asz_nsns, asd_nsns, asl_nsns], ax[1], type_list2,
         return_raw_label('NSNS'))
plot_ecc(x, [amz_nsbh, amd_nsbh, aml_nsbh], [asz_nsbh, asd_nsbh, asl_nsbh], ax[2], type_list2,
         return_raw_label('NSBH'))
plot_ecc(x, [amz_bhns, amd_bhns, aml_bhns], [asz_bhns, asd_bhns, asl_bhns], ax[3], type_list2,
         return_raw_label('BHNS'))

f.legend(loc='upper center', ncols=3)
[i.set_yscale('log') for i in ax]
plt.xlabel('Number of galaxies')
plt.tight_layout()
f.subplots_adjust(top=0.96)
plt.savefig('semi_major_axis_comparison.pdf')
plt.close()
