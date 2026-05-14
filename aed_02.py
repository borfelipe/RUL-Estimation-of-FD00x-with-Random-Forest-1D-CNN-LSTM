from importacao_01 import importar_dados, plots_design ##teste de branch
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import joblib

# =====================================================================
# 1. METADADOS E CONSTANTES
# =====================================================================
COLUNAS_OP = ['Altitude', 'Mach', 'TRA']
CORES_CLUSTER = ['#1F77B4', '#D62728', '#2CA02C', '#9467BD', '#FF7F0E', '#3A3A3A']
CMAP_CLUSTERS = ListedColormap(CORES_CLUSTER)

INFO_SENSORES = {
    "T2": {"nome": "Temp. entrada ventilador", "id": "S1: T_2", "unid": "°R"},
    "T24": {"nome": "Temp. saída LPC", "id": "S2: T_24", "unid": "°R"},
    "T30": {"nome": "Temp. saída HPC", "id": "S3: T_30", "unid": "°R"},
    "T50": {"nome": "Temp. saída LPT", "id": "S4: T_50", "unid": "°R"},
    "P2": {"nome": "Pressão entrada ventilador", "id": "S5: P_2", "unid": "psia"},
    "P15": {"nome": "Pressão duto bypass", "id": "S6: P_15", "unid": "psia"},
    "P30": {"nome": "Pressão saída HPC", "id": "S7: P_30", "unid": "psia"},
    "Nf": {"nome": "Vel. física ventilador", "id": "S8: N_f", "unid": "rpm"},
    "Nc": {"nome": "Vel. física núcleo", "id": "S9: N_c", "unid": "rpm"},
    "Epr": {"nome": "Razão de pressão motor", "id": "S10: epr", "unid": "--"},
    "PS30": {"nome": "Pressão estática saída HPC", "id": "S11: Ps30", "unid": "psia"},
    "phi": {"nome": "Razão fluxo comb./Ps30", "id": "S12: phi", "unid": "pps/psi"},
    "NRf": {"nome": "Vel. corrigida ventilador", "id": "S13: NRf", "unid": "rpm"},
    "NRc": {"nome": "Vel. corrigida núcleo", "id": "S14: NRc", "unid": "rpm"},
    "BPR": {"nome": "Razão de bypass", "id": "S15: BPR", "unid": "--"},
    "farB": {"nome": "Eficiência do queimador", "id": "S16: farB", "unid": "--"},
    "htBleed": {"nome": "Entalpia de sangria", "id": "S17: htBleed", "unid": "--"},
    "Nf_dmd": {"nome": "Vel. ventilador demandada", "id": "S18: Nf_dmd", "unid": "rpm"},
    "PCNfR_dmd": {"nome": "Vel. corrigida demandada", "id": "S19: PCNfR_dmd", "unid": "rpm"},
    "W31": {"nome": "Sangria resfriamento HPT", "id": "S20: W31", "unid": "lbm/s"},
    "W32": {"nome": "Sangria resfriamento LPT", "id": "S21: W32", "unid": "lbm/s"},
}
SENSORES = list(INFO_SENSORES.keys())

# =====================================================================
# 2. CLUSTERIZAÇÃO E PADRONIZAÇÃO
# =====================================================================

def aplicar_clustering_condicoes(df, n_clusters=6, treinar=True):
    df_cluster = df.copy()
    
    if treinar:
        scaler_op = StandardScaler()
        ops_scaled = scaler_op.fit_transform(df_cluster[COLUNAS_OP])
        
        # Se o usuário passou a string, calcula o valor ótimo
        if n_clusters == "no_clusters_otim":
            print("Buscando número ótimo de clusters (intervalo 2-9)...")
            melhor_score = -1
            no_clusters_otim = 2
            
            for k in range(2, 10):
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans_temp.fit_predict(ops_scaled)
                score = silhouette_score(ops_scaled, labels)
                
                if score > melhor_score:
                    melhor_score = score
                    no_clusters_otim = k
            
            print(f"Otimização concluída. Valor ótimo: {no_clusters_otim}")
            n_clusters = no_clusters_otim  # Atribui o valor ótimo à variável final
        
        # Executa o KMeans com o valor definido (manual ou calculado)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cluster['cluster_op'] = kmeans.fit_predict(ops_scaled)
        
        # Salva o modelo e a quantidade de clusters usada
        joblib.dump((scaler_op, kmeans, n_clusters), "modelo_clustering.pkl")
    else:
        scaler_op, kmeans, _ = joblib.load("modelo_clustering.pkl")
        ops_scaled = scaler_op.transform(df_cluster[COLUNAS_OP])
        df_cluster['cluster_op'] = kmeans.predict(ops_scaled)
        
    return df_cluster

def padronizar_por_cluster(df, sensores, treinar=True):
    df_padr = df.copy()
    if treinar:
        estatisticas = {}
        for cluster in df_padr['cluster_op'].unique():
            idx = df_padr['cluster_op'] == cluster
            mean = df_padr.loc[idx, sensores].mean()
            std = df_padr.loc[idx, sensores].std().replace(0, 1) 
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
# 3. VISUALIZAÇÃO EXPLORATÓRIA (INTERATIVA)
# =====================================================================
def configurar_grafico(ax, sensor, is_padr=False):
    info = INFO_SENSORES[sensor]
    ax.set_title(f"{info['id']} - {info['nome']}")
    ax.set_xlabel("RUL (Ciclos Restantes)")
    ax.invert_xaxis()
    ax.set_ylabel("Valor Padronizado" if is_padr else f"Valor ({info['unid']})")
    
    # Aplica o design estético de artigo ao invés do grid cru
    plots_design(ax)

def plotar_sensores(df, sensor_input, usar_cluster, is_padr=False):
    lista_plot = SENSORES if sensor_input == 'todos' else [sensor_input]
    cols = 3 if len(lista_plot) > 1 else 1
    rows = (len(lista_plot) + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = [axs] if len(lista_plot) == 1 else axs.flatten()
    
    for i, sens in enumerate(lista_plot):
        if usar_cluster:
            scatter = axs[i].scatter(df['RUL'], df[sens], c=df['cluster_op'], cmap=CMAP_CLUSTERS, vmin=0, vmax=5, alpha=0.5, s=10)
        else:
            axs[i].scatter(df['RUL'], df[sens], alpha=0.5, color='blue', s=10)
        configurar_grafico(axs[i], sens, is_padr)
        
    for j in range(i + 1, len(axs)): fig.delaxes(axs[j]) 
    
    if usar_cluster and len(lista_plot) == 1:
        plt.colorbar(scatter, ax=axs[0], ticks=range(6)).set_label('Cluster Operacional')
        
    plt.tight_layout()
    plt.show()

def menu_exploratorio():
    print("Importando dados...")
    df_train, _, _ = importar_dados()
    df_cluster = aplicar_clustering_condicoes(df_train, treinar=True)
    
    while True:
        print("\n=== ANÁLISE EXPLORATÓRIA ===")
        print("1 - Ver dados originais (Sensores vs RUL)")
        print("2 - Ver condições operacionais de um motor")
        print("3 - Ver dados padronizados (Z-Score por Cluster)")
        print("0 - Sair")
        
        op = input("Escolha: ").strip()
        if op == '0': break
        
        elif op in ['1', '3']:
            df_plot = padronizar_por_cluster(df_cluster, SENSORES, treinar=True) if op == '3' else df_cluster
            print(f"\nSensores: {', '.join(SENSORES)}")
            sensor = input("Digite o nome do sensor ou 'todos': ").strip()
            if sensor not in SENSORES and sensor != 'todos': continue
            
            cluster = input("Colorir por cluster? (s/n): ").strip().lower() == 's'
            plotar_sensores(df_plot, sensor, cluster, is_padr=(op=='3'))
            
        elif op == '2':
            unidade = int(input("Número do motor: "))
            df_motor = df_cluster[df_cluster['unidade'] == unidade]
            cluster = input("Colorir por cluster? (s/n): ").strip().lower() == 's'
            
            # --- Plot 2D Original ---
            fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            for i, cond in enumerate(COLUNAS_OP):
                if cluster:
                    axs[i].scatter(df_motor['ciclo'], df_motor[cond], c=df_motor['cluster_op'], cmap=CMAP_CLUSTERS, alpha=0.8)
                else:
                    axs[i].plot(df_motor['ciclo'], df_motor[cond], color='black')
                axs[i].set_ylabel(cond)
                plots_design(axs[i]) 
            
            axs[2].set_xlabel("Ciclo")
            plt.suptitle(f"Condições Operacionais (2D) - Motor {unidade}")
            plt.tight_layout()
            plt.show()
            
            # --- Plot 3D Adicionado ---
            ver_3d = input("Gerar gráfico 3D das condições operacionais? (s/n): ").strip().lower() == 's'
            if ver_3d:
                fig3d = plt.figure(figsize=(9, 7))
                ax3d = fig3d.add_subplot(111, projection='3d')
                
                if cluster:
                    scatter3d = ax3d.scatter(df_motor['Altitude'], df_motor['Mach'], df_motor['TRA'], 
                                             c=df_motor['cluster_op'], cmap=CMAP_CLUSTERS, s=20, alpha=0.9)
                    fig3d.colorbar(scatter3d, ax=ax3d, label='Cluster Operacional', pad=0.1)
                else:
                    ax3d.scatter(df_motor['Altitude'], df_motor['Mach'], df_motor['TRA'], 
                                 color='blue', s=20, alpha=0.9)
                
                ax3d.set_title(f"Condições de Operação vs. RUL (3D) - Motor {unidade}")
                ax3d.set_xlabel("Altitude")
                ax3d.set_ylabel("Mach")
                ax3d.set_zlabel("TRA")
                
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    menu_exploratorio()