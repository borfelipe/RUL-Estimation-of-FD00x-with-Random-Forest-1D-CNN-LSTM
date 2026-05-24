import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.image as mpimg
import visualkeras

from A_importacao import plots_design
from C_tratamento import preparar_dados_redes_neurais, obter_dados_tratados_2d

def obter_info_teste():
    """Retorna um DataFrame com unidade, ciclo máximo e RUL real do conjunto de teste"""
    from A_importacao import importar_dados
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
    """Calcula RMSE e Score NASA para um modelo específico"""
    d = df[col_predicao] - df['RUL_true']
    rmse = np.sqrt(np.mean(d**2))
    score = np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1))
    
    return rmse, score

def visualizar_random_forest(rf_model):
    print("\n--- Visualização da Random Forest ---")
    try:
        total_arvores = len(rf_model.estimators_)
        while True:
            escolha = input(f"Qual árvore deseja visualizar (1-{total_arvores}) ou '0' para sair: ").strip()
            if escolha == '0':
                return
            if escolha.isdigit():
                idx = int(escolha) - 1
                if 0 <= idx < total_arvores:
                    break
            print("Entrada inválida. Por favor, digite um número válido.")
            
        arvore_selecionada = rf_model.estimators_[idx]
        feature_names = ['FS', 'DF'] 
        
        print("Gerando o gráfico da árvore... Isso pode levar alguns segundos dependendo da profundidade.")
        plt.figure(figsize=(24, 12))
        plot_tree(arvore_selecionada, 
                  feature_names=feature_names, 
                  filled=True, 
                  rounded=True,
                  fontsize=10)
        
        plt.title(f"Visualização da Árvore {idx + 1} da Random Forest", fontsize=16)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erro ao visualizar a Random Forest: {e}")

def visualizar_cnn(cnn_model, nome_arquivo):
    print("\n--- Visualização da CNN ---")
    nome_imagem = input("Qual nome deseja dar à figura da arquitetura gerada? (Pressione Enter para usar 'arquitetura_cnn.png'): ").strip()
    if not nome_imagem:
        nome_imagem = 'arquitetura_cnn.png'
    if not nome_imagem.endswith('.png'):
        nome_imagem += '.png'

    try:
        print(f"Gerando diagrama e salvando em '{nome_imagem}'...")
        tf.keras.utils.plot_model(cnn_model, 
                                  to_file=nome_imagem, 
                                  show_shapes=True, 
                                  show_layer_names=True, 
                                  rankdir='TB',
                                  dpi=200)
        print(f"Sucesso! Imagem salva como '{nome_imagem}'.")
        
        visualkeras.layered_view(cnn_model, legend=True, to_file='arquitetura_3d.png').show()
        try:
            img = mpimg.imread(nome_imagem)
            plt.figure(figsize=(10, 14))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Arquitetura da CNN", fontsize=16)
            plt.show()
        except:
            pass 
    except ImportError as e:
        print(f"Erro de dependência: Certifique-se de ter o 'pydot' e o 'graphviz' instalados.\nDetalhes: {e}")
    except Exception as e:
        print(f"Erro ao visualizar a CNN: {e}")

def menu_avaliacao(df, model_rf, model_cnn):
    """ df: DataFrame com ['unidade', 'max_ciclo', 'RUL_true', 'RUL_CNN', 'RUL_RF'] """
    while True:
        print("\n=== DASHBOARD DE PROGNÓSTICO FINAL ===")
        print("1 - Exportar TXT")
        print("2 - Gráfico de Diferença (Real vs Previsto)")
        print("3 - Gráfico de Faixa de Erro (Error Bands)")
        print("4 - Histograma de Erros (Distribuição)")
        print("5 - Métricas Globais (RMSE e Score)")
        print("6 - Visualizar árvore da Random Forest")
        print("7 - Visualizar arquitetura da CNN")
        print("8 - Otimização de Custos e Risco")
        print("0 - Sair")
        
        op = input("Escolha: ").strip()
        if op == '0': break

        elif op == '1':
            rmse_rf, score_rf = calcular_metricas(df, 'RUL_RF')
            rmse_cnn, score_cnn = calcular_metricas(df, 'RUL_CNN')

            df_export = df[['unidade', 'RUL_RF', 'RUL_CNN', 'RUL_true']].rename(columns={'RUL_true': 'RUL_real'})

            with open(pred_txt, "w", encoding="utf-8") as f:
                f.write("========================================================\n")
                f.write("METRICAS GLOBAIS DE DESEMPENHO\n")
                f.write("========================================================\n")
                f.write(f"RANDOM FOREST: RMSE: {rmse_rf:.2f} | Score NASA: {score_rf:.2f}\n")
                f.write(f"CNN/LSTM:      RMSE: {rmse_cnn:.2f} | Score NASA: {score_cnn:.2f}\n")
                f.write("========================================================\n\n")
                f.write("TABELA DE PREDICOES POR MOTOR:\n")
                df_export.to_csv(f, sep='\t', index=False)

            print(f"Relatório completo salvo com sucesso em '{pred_txt}'.")

        elif op == '2':
            try:
                ini, fim = map(int, input("Intervalo de motores (ex: 1-100): ").split('-'))
                print("1 - Random Forest\n2 - CNN\n3 - Ambos")
                modelo_op = input("Escolha o que deseja plotar: ").strip()
                
                df_p = df[(df['unidade'] >= ini) & (df['unidade'] <= fim)]
                fig, ax = plt.subplots()
                x = df_p['unidade'].astype(int)
                
                if modelo_op in ['1', '2']:
                    col_pred = 'RUL_RF' if modelo_op == '1' else 'RUL_CNN'
                    label_pred = 'Random Forest' if modelo_op == '1' else 'CNN'
                    diff = df_p[col_pred] - df_p['RUL_true']
                    ax.bar(x, diff, color='#0072BD', width=0.8, label=f'Diferença ({label_pred})')
                
                elif modelo_op == '3':
                    largura = 0.4
                    diff_rf = df_p['RUL_RF'] - df_p['RUL_true']
                    diff_cnn = df_p['RUL_CNN'] - df_p['RUL_true']
                    ax.bar(x - largura/2, diff_rf, width=largura, color='coral', label='Diferença (RF)')
                    ax.bar(x + largura/2, diff_cnn, width=largura, color='royalblue', label='Diferença (CNN)')
                else:
                    print("Opção inválida.")
                    continue
                
                ax.set_xlabel("Engine")
                ax.set_ylabel("Difference Between Predicted and Actual RUL")
                ax.axhline(0, color='black', linewidth=0.8) 
                
                plots_design(ax, fig, tamanho_figura=1, posicao_legenda='fora')
                plt.show()
            except Exception as e: 
                print(f"Erro no intervalo ou visualização: {e}")

        elif op == '3':
            print("1 - Random Forest\n2 - CNN")
            modelo_op = input("Escolha o modelo: ").strip()
            if modelo_op not in ['1', '2']: continue
            col_pred = 'RUL_RF' if modelo_op == '1' else 'RUL_CNN'
            
            # Ordenar pelo RUL real para a curva ficar contínua
            df_sorted = df.sort_values(by='RUL_true').reset_index(drop=True)
            x = np.arange(len(df_sorted))
            y_true = df_sorted['RUL_true']
            y_pred = df_sorted[col_pred]
            
            fig, ax = plt.subplots()
            
            # Faixas de erro preenchidas
            ax.fill_between(x, y_true - 30, y_true + 30, color='lightcoral', alpha=0.5, label='Error band $\pm 30$')
            ax.fill_between(x, y_true - 20, y_true + 20, color='khaki', alpha=0.8, label='Error band $\pm 20$')
            ax.fill_between(x, y_true - 10, y_true + 10, color='lightgreen', alpha=0.8, label='Error band $\pm 10$')
            
            # RUL Real vs Previsto
            ax.plot(x, y_true, color='red', linewidth=1.2, label='Actual RUL')
            ax.scatter(x, y_pred, marker='v', facecolors='none', edgecolors='black', label='Prediction RUL', s=25)
            
            ax.set_xlabel("Test engine units (sorted)")
            ax.set_ylabel("RUL(cycles)")
            plots_design(ax, fig, tamanho_figura=2, posicao_legenda='fora')
            plt.show()

        elif op == '4':
            print("1 - Random Forest\n2 - CNN")
            modelo_op = input("Escolha o modelo: ").strip()
            if modelo_op not in ['1', '2']: continue
            col_pred = 'RUL_RF' if modelo_op == '1' else 'RUL_CNN'
            
            erro = df[col_pred] - df['RUL_true']
            
            fig, ax = plt.subplots()
            # Definimos os limites dos bins explicitamente para garantir passos de 10 em 10
            min_err = int(np.floor(erro.min() / 10) * 10)
            max_err = int(np.ceil(erro.max() / 10) * 10)
            bins_estrutura = np.arange(min_err, max_err + 10, 10)
            
            counts, bins, patches = ax.hist(erro, bins=bins_estrutura, edgecolor='black', alpha=0.8, color='#4A90E2')
            
            # Alinha os xticks com a borda esquerda de cada barra (bin)
            ax.set_xticks(bins)
            
            # Adiciona os números centralizados no topo de cada barra correspondente
            ax.bar_label(patches, padding=2)
            
            ax.set_xlabel("Error = RUL_predict - RUL_actual")
            ax.set_ylabel("Number of times")
            
            plots_design(ax, fig, tamanho_figura=2)
            plt.show()

        elif op == '5':
            rmse_rf, score_rf = calcular_metricas(df, 'RUL_RF')
            rmse_cnn, score_cnn = calcular_metricas(df, 'RUL_CNN')
            
            print("\n" + "="*40)
            print("--- RESULTADOS RANDOM FOREST ---")
            print(f"RMSE: {rmse_rf:.2f} | Score NASA: {score_rf:.2f}")
            print("\n--- RESULTADOS CNN/LSTM ---")
            print(f"RMSE: {rmse_cnn:.2f} | Score NASA: {score_cnn:.2f}")
            print("="*40)

        elif op == '6':
            visualizar_random_forest(model_rf)
            
        elif op == '7':
            visualizar_cnn(model_cnn, nome_cnn)
        
        elif op == '8':
            print("1 - Random Forest\n2 - CNN")
            modelo_op = input("Escolha o modelo para a Otimização de Custos: ").strip()
            if modelo_op not in ['1', '2']: 
                print("Opção inválida.")
                continue
                
            col_pred = 'RUL_RF' if modelo_op == '1' else 'RUL_CNN'
            
            # Cálculo do erro: RUL_predict - RUL_actual
            erro = df[col_pred] - df['RUL_true']
            
            # --- Parâmetros de Custo Otimizados (Modelo Híbrido PHM) ---
            C_p = 3.5e6  # Custo fixo da manutenção preventiva
            C_f = 4.0e6 # Custo fixo da falha inesperada em voo
            C_w = 2.0e4   # Penalidade financeira por ciclo útil desperdiçado (Age cedo demais)
            
            # Limites Espelhados
            margem_min = -erro.max() - 1 
            margem_max = -erro.min() + 1
            N_motores = len(erro)
            
            x_vals = np.linspace(margem_min, margem_max, 200)
            
            c1_vals = []
            c2_vals = []
            ctot_vals = []
            prob_vals = []
            
            for x in x_vals:
                falhou = erro >= -x
                prevenido = erro < -x
                
                # C1: Custo fixo + Desperdício (Quanto mais à esquerda no gráfico, maior o desperdício)
                ciclos_desperdicados = -erro[prevenido] - x
                c1 = np.sum(prevenido) * C_p + np.sum(ciclos_desperdicados) * C_w
                
                # C2: Custo exclusivo dos motores que falharam
                c2 = np.sum(falhou) * C_f
                
                c1_vals.append(c1)
                c2_vals.append(c2)
                ctot_vals.append(c1 + c2)
                prob_vals.append(np.sum(falhou) / N_motores)
                
            # Encontrar o ponto ótimo
            min_idx = np.argmin(ctot_vals)
            opt_x = x_vals[min_idx]
            opt_ctot = ctot_vals[min_idx]
            opt_prob = prob_vals[min_idx]
            
            # --- Gráfico 1: Otimização de Custos ---
            fig1, ax1 = plt.subplots()
            
            ax1.plot(x_vals, c1_vals, label='C1 (Prevenção + Desperdício)', color='green', alpha=0.8)
            ax1.plot(x_vals, c2_vals, label='C2 (Custo de Falha)', color='red', alpha=0.8)
            ax1.plot(x_vals, ctot_vals, label='C_total (Soma)', color='black', linewidth=2)
            
            ax1.axvline(opt_x, color='blue', linestyle='--', 
                        label=f'Ação Ótima (x={opt_x:.1f})\nCusto Mín={opt_ctot:.0f}')
            
            ax1.set_xlabel("Margem de Decisão (Ciclos após RUL Predito)")
            ax1.set_ylabel("Custo Financeiro Relativo")
            ax1.set_xlim(margem_min, margem_max)
            
            plots_design(ax1, fig1, tamanho_figura=2, posicao_legenda='dentro')
            plt.show()
            
            # --- Gráfico 2: Probabilidade de Falha Operacional ---
            fig2, ax2 = plt.subplots()
            
            ax2.plot(x_vals, prob_vals, label='Curva de Risco', color='purple', linewidth=2)
            
            ax2.axvline(opt_x, color='blue', linestyle='--', label=f'Ação Ótima (x={opt_x:.1f})')
            ax2.axhline(opt_prob, color='gray', linestyle=':', 
                        label=f'Risco Aceitável = {opt_prob:.2%}')
            
            ax2.set_xlabel("Margem de Decisão (Ciclos após RUL Predito)")
            ax2.set_ylabel("Probabilidade de Falha")
            ax2.set_xlim(margem_min, margem_max)
            ax2.set_ylim(0, 1)
            
            plots_design(ax2, fig2, tamanho_figura=2, posicao_legenda='dentro')
            plt.show()
                
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
    
    # 1. Predição CNN
    df_info['RUL_CNN'] = model_cnn.predict(X_test_cnn, verbose=0).flatten()

    # 2. Predição Random Forest
    preds_rf_totais = model_rf.predict(X_test_rf)
    df_test_rf['RUL_RF_temp'] = preds_rf_totais
    rul_rf_final = df_test_rf.groupby('unidade')['RUL_RF_temp'].last().values
    df_info['RUL_RF'] = rul_rf_final

    # E. Iniciar Interface
    print("\n✓ Pipeline concluído!")
    menu_avaliacao(df_info, model_rf, model_cnn)