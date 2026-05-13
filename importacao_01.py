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

def plots_design(ax, fig=None, tamanho_figura=1, posicao_legenda='dentro'):
    """Aplica design estético sem alterar dados. Recebe 'ax' obrigatoriamente."""
    # --- CONFIGURAÇÕES DE GEOMETRIA ---
    text_width_latex_cm = 15.0 
    subfigure_fraction = 1.0 if tamanho_figura == 1 else 0.48
    picturewidth_cm = text_width_latex_cm * subfigure_fraction
    hw_ratio = 0.65 
    fontsize = 12
    height_inch = (picturewidth_cm * hw_ratio) / 1.54 # Conversão para polegada

    # --- CONFIGURAÇÃO VISUAL ---
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": 10,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize
    })

    # --- AJUSTES DE EIXOS E GRADE ---
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Ajuste da Moldura (Remove topo e direita)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- LÓGICA DA LEGENDA ---
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        if posicao_legenda == 'dentro':
            ax.legend(loc='upper left', frameon=False, borderaxespad=1.0)
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    # --- AJUSTE FINO DE LAYOUT ---
    if fig is not None:
        fig.tight_layout(pad=0.5)

    return fig