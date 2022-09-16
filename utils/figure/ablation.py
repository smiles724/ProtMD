import matplotlib.pyplot as plt


def ablation_plot(show=False, fontsize=14):
    n_proteins = [10, 20, 30, 40, 50, 64]
    rmse_linear = [3.605, 3.588, 3.312, 2.952, 1.426, 1.413]
    rmse_fine = [1.474, 1.456, 1.436, 1.401, 1.378, 1.367]
    pearson_linear = [-0.503, -0.500, -0.481, - 0.211, 0.556, 0.572]
    pearson_fine = [0.555, 0.562, 0.578, 0.584, 0.595, 0.601]
    spearman_linear = [-0.506, -0.503, -0.483, -0.243, 0.548, 0.569]
    spearman_fine = [0.549, 0.556, 0.567, 0.572, 0.576, 0.587]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.rcParams.update({'font.size': fontsize})

    # fig.suptitle('Horizontally stacked subplots')
    ax1.set_title('Linear-probing')
    ax1.set_xlabel('Number of Proteins in Pre-training', fontsize=fontsize)
    ax1.set_ylabel('RMSE', fontsize=fontsize)
    ax1.plot(n_proteins, rmse_linear, marker="o", label='RMSE', color='tab:red')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Correlations', fontsize=fontsize)
    ax2.plot(n_proteins, pearson_linear, marker="v", label='Pearson', color='tab:blue')
    ax2.plot(n_proteins, spearman_linear, marker="^", label='Spearman', color='tab:green')
    plt.legend()
    ax2.tick_params(axis='y')

    ax3.set_title('Fine-tuning')
    ax3.set_xlabel('Number of Proteins in Pre-training', fontsize=fontsize)
    ax3.set_ylabel('RMSE', fontsize=fontsize)
    ax3.plot(n_proteins, rmse_fine, marker="o", label='RMSE', color='tab:red')
    ax3.tick_params(axis='y')
    ax4 = ax3.twinx()
    ax4.set_ylabel('Correlations', fontsize=fontsize)
    ax4.plot(n_proteins, pearson_fine, marker="v", label='Pearson', color='tab:blue')
    ax4.plot(n_proteins, spearman_fine, marker="^", label='Spearman', color='tab:green')
    plt.legend()
    ax4.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if show:
        plt.show()
    else:
        plt.savefig(f'../../../../../ablation.pdf', bbox_inches='tight')


if __name__ == '__main__':
    ablation_plot()