import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FEATURES = [
    "unidade", "ciclo", "Altitude", "Mach", "TRA",
    "T2", "T24", "T30", "T50", "P2", "P15", "P30",
    "Nf", "Nc", "Epr", "PS30", "phi", "NRf", "NRc",
    "BPR", "farB", "htBleed", "Nfdmd", "PCNfRdmd",
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

def plots_design(ax, fig=None, tamanho_figura=1, posicao_legenda='dentro', is_grade=False, espacamento_ticks=None, labelpad_3d=5):
    # --- DIMENSIONAMENTO ---
    larg_rel, hw_ratio = {1: (1.0, 0.4), 2: (0.48, 0.6), 3: (0.32, 0.6)}.get(tamanho_figura, (1.0, 0.4))
    fontsize = 8
    is_3d = hasattr(ax, 'zaxis')
    
    if is_3d and tamanho_figura == 1:
        hw_ratio = 0.75 

    if fig and not is_grade:
        width_inch = (15.0 * larg_rel) / 2.54 
        fig.set_size_inches(width_inch, width_inch * hw_ratio)

    # --- CONFIGURAÇÃO VISUAL GLOBAL ---
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": fontsize,
        "axes.labelsize": fontsize, "legend.fontsize": fontsize - 2,
        "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
        "axes.linewidth": 0.6,
        "axes.labelpad": 0.5 if not is_3d else 0
    })

    if not is_3d:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.margins(x=0.01, y=0.02)
        for spine in ['top', 'right']: 
            ax.spines[spine].set_visible(False)
            
    else:
        # Posição da câmera (Z à esquerda) e painéis limpos
        ax.view_init(elev=25, azim=135)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        
        # Controle separado de Labelpad para o 3D
        ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = labelpad_3d
        
        # Rotação independente dos eixos (X/Y horizontais, Z vertical)
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        
        # Reatribui os labels forçando a rotação exata
        ax.set_xlabel(ax.get_xlabel(), rotation=0)
        ax.set_ylabel(ax.get_ylabel(), rotation=0)
        ax.set_zlabel(ax.get_zlabel(), rotation=90)
        
        try: ax.set_box_aspect(None, zoom=1.10)
        except AttributeError: pass

    # --- CONTROLE DE ESPAÇAMENTO DOS EIXOS (TICKS) ---
    if espacamento_ticks:
        for eixo in ('x', 'y', 'z'):
            if eixo in espacamento_ticks and espacamento_ticks[eixo] and hasattr(ax, f"{eixo}axis"):
                getattr(ax, f"{eixo}axis").set_major_locator(ticker.MultipleLocator(espacamento_ticks[eixo]))

    ax.tick_params(axis='both', which='major', pad=2, length=3)

    # --- LEGENDA ---
    if ax.get_legend_handles_labels()[1]:
        leg_args = {'loc': 'best', 'borderaxespad': 0.5} if posicao_legenda == 'dentro' \
              else {'loc': 'center left', 'bbox_to_anchor': (1.02, 0.5)}
        ax.legend(frameon=False, **leg_args)

    # --- LAYOUT ---
    if fig:
        if is_grade:
            fig.tight_layout(pad=0.5, h_pad=1.5, w_pad=1.0)
        elif not is_3d:
            fig.tight_layout(pad=0.05)
        else:
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    return fig