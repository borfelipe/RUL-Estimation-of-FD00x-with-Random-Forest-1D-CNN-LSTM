import os
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from importacao_01 import importar_dados, plots_design

# =====================================================================
COLUNAS_OP = ['Altitude', 'Mach', 'TRA']
CMAP_CLUSTERS = ListedColormap(['#1F77B4', '#D62728', '#2CA02C', '#9467BD', '#FF7F0E', '#3A3A3A'])

INFO_SENSORES = {
    "T2": {"id": "S1: T_2", "unid": "°R"}, "T24": {"id": "S2: T_24", "unid": "°R"},
    "T30": {"id": "S3: T_30", "unid": "°R"}, "T50": {"id": "S4: T_50", "unid": "°R"},
    "P2": {"id": "S5: P_2", "unid": "psia"}, "P15": {"id": "S6: P_15", "unid": "psia"},
    "P30": {"id": "S7: P_30", "unid": "psia"}, "Nf": {"id": "S8: N_f", "unid": "rpm"},
    "Nc": {"id": "S9: N_c", "unid": "rpm"}, "Epr": {"id": "S10: epr", "unid": "--"},
    "PS30": {"id": "S11: Ps30", "unid": "psia"}, "phi": {"id": "S12: phi", "unid": "pps/psi"},
    "NRf": {"id": "S13: NRf", "unid": "rpm"}, "NRc": {"id": "S14: NRc", "unid": "rpm"},
    "BPR": {"id": "S15: BPR", "unid": "--"}, "farB": {"id": "S16: farB", "unid": "--"},
    "htBleed": {"id": "S17: htBleed", "unid": "--"}, "Nf_dmd": {"id": "S18: Nf_dmd", "unid": "rpm"},
    "PCNfR_dmd": {"id": "S19: PCNfR_dmd", "unid": "rpm"}, "W31": {"id": "S20: W31", "unid": "lbm/s"},
    "W32": {"id": "S21: W32", "unid": "lbm/s"},
}
SENSORES = list(INFO_SENSORES.keys())

# =====================================================================
# CLUSTERIZAÇÃO E PADRONIZAÇÃO
# =====================================================================
def aplicar_clustering_condicoes(df, n_clusters=6, treinar=True):
    df_cluster = df.copy()
    if treinar:
        scaler_op = StandardScaler()
        ops_scaled = scaler_op.fit_transform(df_cluster[COLUNAS_OP])
        
        if n_clusters == "no_clusters_otim":
            melhor_score, n_clusters = -1, 2
            for k in range(2, 10):
                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(ops_scaled)
                score = silhouette_score(ops_scaled, labels)
                if score > melhor_score:
                    melhor_score, n_clusters = score, k
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cluster['cluster_op'] = kmeans.fit_predict(ops_scaled)
        joblib.dump((scaler_op, kmeans, n_clusters), "modelo_clustering.pkl")
    else:
        scaler_op, kmeans, _ = joblib.load("modelo_clustering.pkl")
        df_cluster['cluster_op'] = kmeans.predict(scaler_op.transform(df_cluster[COLUNAS_OP]))
        
    return df_cluster

def padronizar_por_cluster(df, sensores, treinar=True):
    df_padr = df.copy()
    if treinar:
        estatisticas = {}
        for cluster in df_padr['cluster_op'].unique():
            idx = df_padr['cluster_op'] == cluster
            mean, std = df_padr.loc[idx, sensores].mean(), df_padr.loc[idx, sensores].std().replace(0, 1)
            estatisticas[cluster] = {'mean': mean, 'std': std}
            df_padr.loc[idx, sensores] = (df_padr.loc[idx, sensores] - mean) / std
        joblib.dump(estatisticas, "stats_padronizacao_cluster.pkl")
    else:
        estatisticas = joblib.load("stats_padronizacao_cluster.pkl")
        for cluster in df_padr['cluster_op'].unique():
            if cluster in estatisticas:
                idx = df_padr['cluster_op'] == cluster
                df_padr.loc[idx, sensores] = (df_padr.loc[idx, sensores] - estatisticas[cluster]['mean']) / estatisticas[cluster]['std']
    return df_padr

# =====================================================================
# VISUALIZAÇÃO
# =====================================================================
def configurar_grafico(ax, sensor, is_padr=False):
    info = INFO_SENSORES[sensor]
    ax.set_xlabel("RUL")
    ax.invert_xaxis()
    ax.set_ylabel("Valor Padronizado" if is_padr else f"{info['id']} ({info['unid']})")

def plotar_sensores(df, usar_cluster, is_padr=False, modo_visualizacao='individual'):
    os.makedirs('imagens', exist_ok=True)
    sufixo = "-NF" if is_padr else ""
    tamanho = 3
    
    if modo_visualizacao == 'grade':
        cols = 3
        rows = (len(SENSORES) + cols - 1) // cols
        
        base_w_inch = (15.0 * 0.32) / 2.54 
        fig, axs = plt.subplots(rows, cols, figsize=(cols * base_w_inch, rows * base_w_inch))
        axs = axs.flatten()
        
        for i, sens in enumerate(SENSORES):
            c = df['cluster_op'] if usar_cluster else 'blue'
            axs[i].scatter(df['RUL'], df[sens], c=c, cmap=CMAP_CLUSTERS if usar_cluster else None, alpha=0.5, s=0.5)
            configurar_grafico(axs[i], sens, is_padr)
            plots_design(axs[i], tamanho_figura=tamanho)
            
        for j in range(i + 1, len(axs)): fig.delaxes(axs[j])
        
        plots_design(axs[0], fig=fig, tamanho_figura=tamanho, is_grade=True)
        plt.show() 
        
    else:
        for sens in SENSORES:
            fig, ax = plt.subplots()
            c = df['cluster_op'] if usar_cluster else 'blue'
            ax.scatter(df['RUL'], df[sens], c=c, cmap=CMAP_CLUSTERS if usar_cluster else None, alpha=0.5, s=0.75)
            
            configurar_grafico(ax, sens, is_padr)
            plots_design(ax, fig=fig, tamanho_figura=tamanho)
            
            num_y = INFO_SENSORES[sens]['id'].split(':')[0].replace('S', '')
            plt.savefig(f"imagens/FD001-S{num_y}{sufixo}.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
            plt.close(fig)
            
        print(f"\n[+] Imagens individuais (tamanho {tamanho}) guardadas em 'imagens/'.")

def menu_exploratorio():
    print("Importando dados...")
    df_train, _, _ = importar_dados()
    df_cluster = aplicar_clustering_condicoes(df_train, treinar=True)
    
    while True:
        print("\n=== ANÁLISE EXPLORATÓRIA DOS DADOS ===")
        print("1 - Dados originais")
        print("2 - Dados padronizados")
        print("3 - Condições operacionais de um motor")
        print("0 - Sair")
        
        op = input("Escolha: ").strip()
        if op == '0': break
        
        elif op in ['1', '2']:
            is_padr = (op == '2')
            df_plot = padronizar_por_cluster(df_cluster, SENSORES, treinar=True) if is_padr else df_cluster
            cluster = input("Colorir por cluster? (s/n): ").strip().lower() == 's'
            modo = 'grade' if input("Gerar em 'grade' (g) ou 'individual' (i)? (g/i): ").strip().lower() == 'g' else 'individual'
            
            print("Gerando gráficos...")
            plotar_sensores(df_plot, usar_cluster=cluster, is_padr=is_padr, modo_visualizacao=modo)
            
        elif op == '3':
            unidade = int(input("Número do motor: "))
            df_motor = df_cluster[df_cluster['unidade'] == unidade]
            cluster = input("Colorir por cluster? (s/n): ").strip().lower() == 's'
            
            # --- Plot 2D ---
            fig, axs = plt.subplots(3, 1, sharex=True)
            for i, cond in enumerate(COLUNAS_OP):
                if cluster:
                    axs[i].scatter(df_motor['ciclo'], df_motor[cond], c=df_motor['cluster_op'], cmap=CMAP_CLUSTERS, alpha=0.75, s=15)
                else:
                    axs[i].plot(df_motor['ciclo'], df_motor[cond], color='black')
                axs[i].set_ylabel(cond)
                plots_design(axs[i], tamanho_figura=1) 
            
            axs[2].set_xlabel("Ciclo")
            fig.suptitle(f"Condições Operacionais (2D) - Motor {unidade}", y=1.0)
            plots_design(axs[0], fig=fig, tamanho_figura=1)
            plt.show()
            
            # --- Plot 3D ---
            if input("Gerar gráfico 3D? (s/n): ").strip().lower() == 's':
                fig3d = plt.figure()
                ax3d = fig3d.add_subplot(111, projection='3d')
                c = df_motor['cluster_op'] if cluster else 'blue'
                scatter3d = ax3d.scatter(df_motor['Altitude'], df_motor['Mach'], df_motor['TRA'], c=c, cmap=CMAP_CLUSTERS if cluster else None, s=20, alpha=0.75)
                
                if cluster: fig3d.colorbar(scatter3d, ax=ax3d, label='Cluster Operacional', pad=0.1)
                ax3d.set(title=f"Operação vs RUL (3D) - Motor {unidade}", xlabel="Altitude", ylabel="Mach", zlabel="TRA")
                
                plots_design(ax3d, fig=fig3d, tamanho_figura=2)
                plt.show()

if __name__ == "__main__":
    menu_exploratorio()