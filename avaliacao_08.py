import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from importacao_01 import plots_design
from tratamento_03 import preparar_dados_redes_neurais, obter_dados_tratados_2d
import pandas as pd

def obter_info_teste():
    """Retorna um DataFrame com unidade, ciclo máximo e RUL real do conjunto de teste"""
    from importacao_01 import importar_dados
    _, df_test, y_test_final = importar_dados()
    
    info = []
    unidades = df_test['unidade'].unique()
    for i, unid in enumerate(unidades):
        max_c = df_test[df_test['unidade'] == unid]['ciclo'].max()
        info.append({
            'unidade': int(unid),
            'max_ciclo': int(max_c),
            'RUL_true': int(y_test_final[i])
        })
    return pd.DataFrame(info)


pred_txt = 'FD00x_predicoes.txt'
tw = 30 # time window lenght
nome_rf = 'FD00x_rf.joblib'
nome_cnn = 'FD00x_cnn.keras'

# =====================================================================
# 1. FUNÇÕES DE APOIO E MÉTRICAS
# =====================================================================

def calcular_metricas(df, col_predicao):
    """Calcula RMSE, Score NASA e RA (Relative Accuracy) Médio para um modelo específico"""
    d = df[col_predicao] - df['RUL_true']
    rmse = np.sqrt(np.mean(d**2))
    score = np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1))
    
    # RA: 1 - (|Erro| / RUL_Real). Limitado a 0 para não distorcer médias.
    ra_array = 100 * np.maximum(0, 1 - np.abs(d) / np.maximum(df['RUL_true'], 1))
    ra_medio = np.mean(ra_array)
    
    return rmse, score, ra_medio

def menu_avaliacao(df):
    """ df: DataFrame com ['unidade', 'max_ciclo', 'RUL_true', 'RUL_CNN', 'RUL_RF'] """
    while True:
        print("\n=== DASHBOARD DE PROGNÓSTICO FINAL ===")
        print("1 - Exportar TXT")
        print("2 - Gráfico Real vs Previsto (Sobreposto)")
        print("3 - Métricas Globais (RMSE, Score, RA)")
        print("4 - Análise de RA (Relative Accuracy) e CV (Convergence)")
        print("0 - Sair")
        
        op = input("Escolha: ").strip()
        if op == '0': break

        elif op == '1':
            # 1. Calcula as métricas (mesma lógica da opção 3)
            rmse_rf, score_rf, ra_rf = calcular_metricas(df, 'RUL_RF')
            rmse_cnn, score_cnn, ra_cnn = calcular_metricas(df, 'RUL_CNN')

            # 2. Prepara o DataFrame para exportação
            df_export = df[['unidade', 'RUL_RF', 'RUL_CNN', 'RUL_true']].rename(columns={'RUL_true': 'RUL_real'})

            # 3. Grava o arquivo misturando texto e tabela
            nome_arquivo = pred_txt
            with open(nome_arquivo, "w", encoding="utf-8") as f:
                # Escreve o cabeçalho de métricas
                f.write("========================================================\n")
                f.write("METRICAS GLOBAIS DE DESEMPENHO\n")
                f.write("========================================================\n")
                f.write(f"RANDOM FOREST: RMSE: {rmse_rf:.2f} | Score NASA: {score_rf:.2f} | RA: {ra_rf:.2f}%\n")
                f.write(f"CNN/LSTM:      RMSE: {rmse_cnn:.2f} | Score NASA: {score_cnn:.2f} | RA: {ra_cnn:.2f}%\n")
                f.write("========================================================\n\n")
                f.write("TABELA DE PREDICOES POR MOTOR:\n")
                
                # Salva o DataFrame logo abaixo usando o arquivo 'f' já aberto
                df_export.to_csv(f, sep='\t', index=False)

            print(f"Relatório completo salvo com sucesso em '{nome_arquivo}'.")

        elif op == '2':
            try:
                ini, fim = map(int, input("Intervalo de motores (ex: 1-25): ").split('-'))
                print("1 - Random Forest\n2 - CNN\n3 - Ambos")
                modelo_op = input("Escolha o que deseja plotar: ").strip()
                
                df_p = df[(df['unidade'] >= ini) & (df['unidade'] <= fim)]
                fig, ax = plt.subplots(figsize=(14, 6))
                x = np.arange(len(df_p))
                
                if modelo_op in ['1', '2']:
                    col_pred = 'RUL_RF' if modelo_op == '1' else 'RUL_CNN'
                    label_pred = 'RUL Random Forest' if modelo_op == '1' else 'RUL CNN'
                    
                    # Fundo: Real
                    ax.bar(x, df_p['RUL_true'], color='gray', alpha=0.8, label='RUL Real', width=0.6)
                    # Frente: Predito (Verde/Vermelho)
                    cores = ['green' if p < r else 'red' for p, r in zip(df_p[col_pred], df_p['RUL_true'])]
                    ax.bar(x, df_p[col_pred], color=cores, alpha=0.4, label=label_pred, width=0.6)
                
                elif modelo_op == '3':
                    largura = 0.35
                    # Fundo: Real bem largo
                    ax.bar(x, df_p['RUL_true'], color='gray', alpha=0.4, label='RUL Real', width=0.8)
                    # Barras lado a lado para comparação direta
                    ax.bar(x - largura/2, df_p['RUL_RF'], width=largura, color='coral', alpha=0.9, label='Predito (RF)')
                    ax.bar(x + largura/2, df_p['RUL_CNN'], width=largura, color='royalblue', alpha=0.9, label='Predito (CNN)')
                else:
                    print("Opção inválida.")
                    continue
                
                ax.set_xticks(x)
                ax.set_xticklabels(df_p['unidade'].astype(int))
                ax.set_ylabel("RUL (Ciclos)")
                ax.set_title("Comparação: RUL Real vs Previsto")
                plots_design(ax, fig, posicao_legenda='fora')
                plt.show()
            except Exception as e: 
                print(f"Erro no intervalo ou visualização: {e}")

        elif op == '3':
            rmse_rf, score_rf, ra_rf = calcular_metricas(df, 'RUL_RF')
            rmse_cnn, score_cnn, ra_cnn = calcular_metricas(df, 'RUL_CNN')
            
            print("\n" + "="*40)
            print("--- RESULTADOS RANDOM FOREST ---")
            print(f"RMSE: {rmse_rf:.2f} | Score NASA: {score_rf:.2f} | RA Médio: {ra_rf:.2f}%")
            print("\n--- RESULTADOS CNN/LSTM ---")
            print(f"RMSE: {rmse_cnn:.2f} | Score NASA: {score_cnn:.2f} | RA Médio: {ra_cnn:.2f}%")
            print("="*40)

        elif op == '4':
            try:
                print("1 - Random Forest\n2 - CNN")
                modelo_op = input("Escolha qual modelo analisar: ").strip()
                if modelo_op not in ['1', '2']:
                    print("Opção inválida.")
                    continue
                
                col_pred = 'RUL_RF' if modelo_op == '1' else 'RUL_CNN'
                nome_modelo = 'Random Forest' if modelo_op == '1' else 'CNN'

                # Prepara os dados de erro e RA focando no modelo escolhido
                d = df[col_pred] - df['RUL_true']
                df['RA'] = 100 * np.maximum(0, 1 - np.abs(d) / np.maximum(df['RUL_true'], 1))
                df['Erro'] = np.abs(d)

                fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f"Análise de Desempenho - {nome_modelo}", fontsize=14, fontweight='bold')
                
                # Plot 1: RA
                axs[0].bar(df['unidade'], df['RA'], color='royalblue', alpha=0.7)
                axs[0].set_title("Relative Accuracy (RA) por Motor")
                axs[0].set_xlabel("Motor (ID)")
                axs[0].set_ylabel("RA (%)")
                axs[0].set_ylim(0, 105)
                plots_design(axs[0], posicao_legenda='dentro')

                # Plot 2: CV
                axs[1].scatter(df['RUL_true'], df['Erro'], color='darkorange', alpha=0.6, edgecolors='k')
                axs[1].invert_xaxis() 
                axs[1].set_title("Convergence (CV): Erro vs RUL Restante")
                axs[1].set_xlabel("RUL Real (Aproximando da Falha $\\rightarrow$)")
                axs[1].set_ylabel("Erro Absoluto |Predito - Real|")
                plots_design(axs[1], posicao_legenda='dentro')

                plt.tight_layout()
                plt.show()
            except Exception as e: 
                print(f"Erro na visualização: {e}")

# =====================================================================
# 2. PIPELINE DE EXECUÇÃO
# =====================================================================

if __name__ == "__main__":
    print("Iniciando pipeline de predição combinada...")
    
    # A. Preparar Dados para Redes Neurais (CNN) e carregar df_info base
    print("- Processando dados para CNN...")
    _, _, X_test_cnn = preparar_dados_redes_neurais(seq_len=tw)
    df_info = obter_info_teste() 

    # B. Preparar Dados para Random Forest (2D)
    print("- Processando dados para Random Forest...")
    estagios = obter_dados_tratados_2d()
    df_test_rf = estagios['fs_test']
    X_test_rf = df_test_rf[['FS', 'DF']].values

    # C. Carregar Modelos
    print("- Carregando modelos salvos...")
    model_cnn = tf.keras.models.load_model(nome_cnn, compile=False)
    model_cnn.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='mse')
    
    model_rf = joblib.load(nome_rf)

  # D. Predição Conjunta
    print("- Realizando predições...")
    
    # 1. Predição CNN (Já deve estar retornando 100 valores)
    df_info['RUL_CNN'] = model_cnn.predict(X_test_cnn, verbose=0).flatten()

    # 2. Predição Random Forest
    # Fazemos a predição para todos os 13.096 ciclos
    preds_rf_totais = model_rf.predict(X_test_rf)
    
    # Adicionamos essas predições temporariamente ao DataFrame de teste
    df_test_rf['RUL_RF_temp'] = preds_rf_totais
    
    # Agrupamos pelo ID do motor e extraímos apenas a ÚLTIMA predição feita para cada um
    rul_rf_final = df_test_rf.groupby('unidade')['RUL_RF_temp'].last().values
    
    # Agora sim, alocamos os 100 valores finais no df_info
    df_info['RUL_RF'] = rul_rf_final

    # E. Iniciar Interface
    print("\n✓ Pipeline concluído!")
    menu_avaliacao(df_info)

    # E. Iniciar Interface
    print("\n✓ Pipeline concluído!")
    menu_avaliacao(df_info)