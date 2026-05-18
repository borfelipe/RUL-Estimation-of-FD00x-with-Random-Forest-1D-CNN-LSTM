import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    "unidade", "ciclo", "Altitude", "Mach", "TRA",
    "T2", "T24", "T30", "T50", "P2", "P15", "P30",
    "Nf", "Nc", "Epr", "PS30", "phi", "NRf", "NRc",
    "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd",
    "W31", "W32"
]

def importar_dados(train_path="train.txt", test_path="test.txt", rul_path="rul.txt"):
    def ler_txt(caminho):
        df = pd.read_csv(caminho, sep=r"\s+", header=None, engine='python').dropna(axis=1, how='all')
        df.columns = FEATURES[:df.shape[1]]
        return df

    df_train = ler_txt(train_path)
    df_train['RUL'] = df_train.groupby('unidade')['ciclo'].transform('max') - df_train['ciclo']

    df_test = ler_txt(test_path)
    df_test['RUL'] = None

    df_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, engine='python').dropna(axis=1, how='all')
    vetor_rul_teste = df_rul[0].values

    return df_train, df_test, vetor_rul_teste

def plots_design(ax, fig=None, tamanho_figura=1, posicao_legenda='dentro', is_grade=False):
    mapa_tamanhos = {
        1: (1.0, 0.4), 
        2: (0.48, 0.8),  
        3: (0.32, 0.6)   
    }
    
    larg_rel, hw_ratio = mapa_tamanhos.get(tamanho_figura, (1.0, 0.4))
    fontsize = 8
    
    is_3d = ax.name == '3d'
    if is_3d and tamanho_figura == 1:
        hw_ratio = 0.75 

    # --- DIMENSIONAMENTO ---
    if fig is not None and not is_grade:
        width_inch = (15.0 * larg_rel) / 2.54 
        fig.set_size_inches(width_inch, width_inch * hw_ratio)

    # --- CONFIGURAÇÃO VISUAL GLOBAL ---
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": fontsize,
        "axes.labelsize": fontsize, "axes.labelpad": 0.5,
        "legend.fontsize": fontsize - 2,
        "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
        "axes.linewidth": 0.8 
    })

    # --- EIXOS, MARGENS E GRADE ---
    if not is_3d:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        for spine in ['top', 'right']: 
            ax.spines[spine].set_visible(False)
        ax.margins(x=0.01, y=0.02)
    else:
        # Estilo mais limpo para os painéis de fundo do 3D
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    ax.tick_params(axis='both', which='major', pad=2, length=3)

    # --- LEGENDA ---
    if ax.get_legend_handles_labels()[1]:
        if posicao_legenda == 'dentro':
            ax.legend(loc='best', frameon=False, borderaxespad=0.5)
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    # --- LAYOUT ---
    if fig is not None:
        if is_grade:
            fig.tight_layout(pad=0.5, h_pad=1.5, w_pad=1.0)
        else:
            fig.tight_layout(pad=0.05) 

    return fig