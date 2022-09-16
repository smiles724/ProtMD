import matplotlib.pyplot as plt


def pdb_plot():
    methods = ['ProtMD', 'IECov', '3DCNN', 'HoloProt', 'MaSIF', 'ProtTrans', 'TAPE', 'LSTM', 'DeepDTA']
    rmse = [1.367, 1.554, 1.429, 1.464, 1.484, 1.544, 1.890, 1.985, 1.565]
    rmse_error = [0.014, 0.016, 0.042, 0.006, 0.018, 0.015, 0.034, 0.016, 0.080]
    r_p = [0.601, 0.414, 0.541, 0.509, 0.467, 0.438, 0.338, 0.165, 0.573]
    r_p_error = [0.036, 0.053, 0.029, 0.00, 0.020, 0.053, 0.044, 0.006, 0.022]
    r_s = [0.587, 0.428, 0.532, 0.500, 0.455, 0.434, 0.286, 0.152, 0.574]
    r_s_error = [0.042, 0.032, 0.033, 0.005, 0.014, 0.058, 0.124, 0.024, 0.024]
    color = ['firebrick'] + ['royalblue'] * (len(rmse) - 1)

    fig, axs = plt.subplots(3, 1)
    axs[0].bar(methods, rmse, yerr=rmse_error, align='center', alpha=0.5, ecolor='black', capsize=10, color=color,
               error_kw=dict(ecolor='grey', lw=2, capsize=5, capthick=2))
    axs[0].text(-0.35, rmse[0] + 0.05, str(rmse[0]), color='black')
    axs[0].text(0, 1.9, 'Binding Affinity Prediction', color='black', fontweight='bold')

    axs[0].set_ylim([1.2, 2.1])
    axs[0].set_ylabel('RMSE')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].bar(methods, r_p, yerr=r_p_error, align='center', alpha=0.5, ecolor='black', capsize=10, color=color,
               error_kw=dict(ecolor='grey', lw=2, capsize=5, capthick=2))
    axs[1].text(-0.35, r_p[0] + 0.05, str(r_p[0]), color='black')

    axs[1].set_ylim([0.1, 0.8])
    axs[1].set_ylabel('Pearson')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    axs[2].bar(methods, r_s, yerr=r_s_error, align='center', alpha=0.5, ecolor='black', capsize=10, color=color,
               error_kw=dict(ecolor='grey', lw=2, capsize=5, capthick=2))
    axs[2].text(-0.35, r_s[0] + 0.05, str(r_s[0]), color='black')

    axs[2].set_ylim([0.1, 0.8])
    axs[2].set_ylabel('Spearman')
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def lep_plot():
    methods = ['ProtMD', '3DCNN', '3DGCN', 'Cormorant', 'DeepDTA']
    roc = [0.742, 0.589, 0.681, 0.663, 0.696]
    roc_error = [0.039, 0.020, 0.062, 0.100, 0.021]
    prc = [0.724, 0.483, 0.598, 0.551, 0.550]
    prc_error = [0.041, 0.037, 0.135, 0.121, 0.024]
    color = ['lightcoral'] + ['orange'] * (len(methods) - 1)

    fig, axs = plt.subplots(2, 1)
    axs[0].bar(methods, roc, yerr=roc_error, align='center', alpha=0.5, ecolor='black', capsize=10, color=color,
               error_kw=dict(ecolor='grey', lw=2, capsize=5, capthick=2))
    axs[0].text(-0.2, roc[0] + 0.05, str(roc[0]), color='black')
    axs[0].text(0.65, 0.8, 'Ligand Efficacy Prediction', color='black', fontweight='bold')

    axs[0].set_ylim([0.55, 0.8])
    axs[0].set_ylabel('AUROC')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].bar(methods, prc, yerr=prc_error, align='center', alpha=0.5, ecolor='black', capsize=10, color=color,
               error_kw=dict(ecolor='grey', lw=2, capsize=5, capthick=2))
    axs[1].text(-0.2, prc[0] + 0.05, str(prc[0]), color='black')

    axs[1].set_ylim([0.4, 0.8])
    axs[1].set_ylabel('AUPRC')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    # plt.tight_layout()
    plt.show()


lep_plot()



