import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pwlf
from tratamento_03 import *
from importacao_01 import *

def menu_visualizacao_tratamento():
    print("Processando pipeline de dados... Aguarde.")
    estagios = obter_dados_tratados_2d()
    cps = estagios["cps"]

    while True:
        print("\n=== DASHBOARD: VISUALIZAÇÃO POR ESTÁGIO ===")
        print("1 - Sensor Normalizado vs Filtrado")
        print("2 - Sinal extraído (Treinamento e Teste)")
        print("3 - Change-Point Detection (PWLF Interativo) e Ajuste de Rótulos RUL")
        print("4 - Relação entre RUL e a Feature de Diferença (Treino e Teste)")
        print("0 - Sair")
        
        op = input("Escolha: ").strip()
        if op == '0': break
        
        if op == '1':
            unid = int(input("Número do motor: "))
            d_norm = estagios["normalizado"][estagios["normalizado"]['unidade'] == unid]
            d_filt = estagios["filtrado"][estagios["filtrado"]['unidade'] == unid]
            
            fig, axs = plt.subplots(7, 2, figsize=(14, 18), sharex=True)
            axs = axs.flatten()
            for i, col in enumerate(COLUNAS_MANTER):
                axs[i].set_xlabel("Ciclo")
                axs[i].set_ylabel(f"Leitura normalizada de {col}")
                axs[i].plot(d_norm['ciclo'], d_norm[col], color='gray', alpha=0.5, label='Original')
                axs[i].plot(d_filt['ciclo'], d_filt[col], color='blue', label='Filtrado')
                axs[i].set_title(col)
                # Correção: Adicionado o parâmetro 'fig' exigido pela sua função
                plots_design(axs[i], fig, posicao_legenda='fora') 
            plt.tight_layout()
            plt.show()

        elif op == '2':
            # Correção: As chaves agora são fs_train e fs_test (Fused Signal)
            df_train = estagios["fs_train"] 
            df_test = estagios["fs_test"]
            
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            
            for _, d in df_train.groupby('unidade'):
                axs[0].plot(d['ciclo'], d['FS'])
                
            axs[0].set_xlabel("Ciclo")
            axs[0].set_ylabel("Sinal Fundido")
            axs[0].set_ylim(-0.05, 1.05) 
            # Correção: Adicionado o parâmetro 'fig'
            plots_design(axs[0], fig, posicao_legenda='dentro') 
            
            for _, d in df_test.groupby('unidade'):
                axs[1].plot(d['ciclo'], d['FS'])
                
            axs[1].set_xlabel("Ciclo")
            axs[1].set_ylabel("Sinal Fundido")
            axs[1].set_ylim(-0.05, 0.8) 
            # Correção: Adicionado o parâmetro 'fig'
            plots_design(axs[1], fig, posicao_legenda='dentro')
            
            plt.tight_layout()
            plt.show()
        
        elif op == '3':
            # 1. Escolha do número de segmentos primeiro
            n_seg = int(input("Número de segmentos (ex: 2 para 1 Change-Point): "))
            modo = input("Deseja visualizar todos os motores (T) ou um específico (E)? ").strip().lower()
            
            fs_data = estagios["fs_train"]
            
            if modo == 't':
                # --- PLOTAGEM DE TODOS OS MOTORES (Dinâmico com n_seg) ---
                print(f"Processando PWLF para todos os motores com {n_seg} segmentos... Aguarde.")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for unit, dados_m in fs_data.groupby('unidade'):
                    # Fit dinâmico para cada motor baseado no n_seg escolhido
                    my_pwlf = pwlf.PiecewiseLinFit(dados_m['ciclo'].values, dados_m['FS'].values)
                    cps_m = my_pwlf.fit(n_seg)
                    cp_principal = cps_m[1]
                    
                    # Cálculo do RUL CPD temporário para o gráfico
                    max_ciclo = dados_m['ciclo'].max()
                    rul_cpd_m = np.where(
                        dados_m['ciclo'] < cp_principal, 
                        max_ciclo - cp_principal, 
                        dados_m['RUL']
                    )
                    
                    ax.plot(dados_m['ciclo'], rul_cpd_m, alpha=0.5, label='_nolegend_')
                
                ax.set_xlabel('Ciclo')
                ax.set_ylabel('RUL (CPD)')
                ax.set_title(f'Rótulos RUL CPD para todos os motores ({n_seg} segmentos)')
                
                plots_design(ax, fig, tamanho_figura=1)
                plt.tight_layout()
                plt.show()
                
            else:
                # --- PLOTAGEM ESPECÍFICA (Lado a Lado) ---
                unid = int(input("Número do motor: "))
                dados = fs_data[fs_data['unidade'] == unid].copy()
                
                if dados.empty:
                    print(f"Motor {unid} não encontrado.")
                else:
                    # PWLF Fit Interativo
                    my_pwlf = pwlf.PiecewiseLinFit(dados['ciclo'].values, dados['FS'].values)
                    cps_encontrados = my_pwlf.fit(n_seg)
                    cp_principal = cps_encontrados[1] 
                    
                    # Cálculo dinâmico do RUL CPD para visualização
                    max_ciclo = dados['ciclo'].max()
                    dados['RUL_CPD_VIS'] = np.where(
                        dados['ciclo'] < cp_principal, 
                        max_ciclo - cp_principal, 
                        dados['RUL']
                    )
                    
                    x_hat = np.linspace(dados['ciclo'].min(), dados['ciclo'].max(), 100)
                    y_hat = my_pwlf.predict(x_hat)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Gráfico da Esquerda: FS + PWLF
                    ax1.plot(dados['ciclo'], dados['FS'], 'k-', alpha=0.5, label='Fused Signal')
                    ax1.plot(x_hat, y_hat, 'b-', linewidth=2, label=f'PWLF ({n_seg} segs)')
                    for cp in cps_encontrados[1:-1]:
                        ax1.axvline(cp, color='r', linestyle='--', alpha=0.7, label=f'CP: {int(cp)}')
                    ax1.set_xlabel('Ciclo')
                    ax1.set_ylabel('Sinal Fundido')
                    ax1.set_title('Detecção de Change-Point (PWLF)')
                    ax1.legend()
                    
                    # Gráfico da Direita: RUL Linear vs CPD
                    ax2.plot(dados['ciclo'], dados['RUL'], 'k--', label='RUL Linear')
                    ax2.plot(dados['ciclo'], dados['RUL_CPD_VIS'], 'b', label=f'RUL CPD')
                    ax2.axvline(cp_principal, color='r', linestyle=':', label='Início Degradação')
                    ax2.set_xlabel('Ciclo')
                    ax2.set_ylabel('RUL')
                    ax2.set_title('Ajuste dos Rótulos RUL')
                    ax2.legend()
                    
                    plots_design(ax1, fig, tamanho_figura=2, posicao_legenda='dentro')
                    plots_design(ax2, fig, tamanho_figura=2, posicao_legenda='dentro')
                    plt.tight_layout()
                    plt.show()

        elif op == '4':
            tipo = input("(C)írculos ou (I)D? ").lower()
            
            # --- 1. DADOS DE TREINO (Truncamento Aleatório pré-calculado) ---
            res_tr = estagios["corr_data_train"]
            corr_tr = estagios["corr_val_train"]
            
            # --- 2. DADOS DE TESTE (Truncamento Natural: Último ponto registrado) ---
            # Como y_test_true não está no estagios, importamos aqui apenas para o plot
            _, _, y_test_true = importar_dados() 
            df_test = estagios["fs_test"]
            
            res_test_list = []
            for i, (unit, data) in enumerate(df_test.groupby('unidade')):
                # Pegamos o valor da DF no último ciclo (que já é |FS_último - FS_primeiro|)
                ultimo_ponto = data.iloc[-1]
                res_test_list.append({
                    'u': unit, 
                    'd': ultimo_ponto['DF'], 
                    'r': y_test_true[i] # RUL real do gabarito
                })
            
            res_ts = pd.DataFrame(res_test_list)
            corr_ts = res_ts['d'].corr(res_ts['r'])
            
            # --- 3. PLOTAGEM COMPARATIVA ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
            
            # Subplot Treino (Aleatório)
            if tipo == 'i':
                # TRUQUE DE AUTO-SCALE: Plota pontos invisíveis para o matplotlib ajustar os eixos corretamente
                ax1.scatter(res_tr['d'], res_tr['r'], alpha=0)
                for _, r in res_tr.iterrows(): 
                    ax1.text(r['d'], r['r'], str(int(r['u'])), color='blue', fontsize=9, ha='center', va='center')
            else: 
                ax1.scatter(res_tr['d'], res_tr['r'], facecolors='none', edgecolors='blue', alpha=0.7)
            
            ax1.set_title(f"Treino: RUL vs Difference Feature\n(Truncamento Aleatório) | Corr: {corr_tr:.4f}")
            ax1.set_xlabel("Difference Feature ")
            ax1.set_ylabel("RUL")
            plots_design(ax1, fig, tamanho_figura=2)
            
            # Subplot Teste (Natural/Final)
            if tipo == 'i':
                # TRUQUE DE AUTO-SCALE: Mesma lógica aplicada ao teste
                ax2.scatter(res_ts['d'], res_ts['r'], alpha=0)
                for _, r in res_ts.iterrows(): 
                    ax2.text(r['d'], r['r'], str(int(r['u'])), color='green', fontsize=9, ha='center', va='center')
            else: 
                ax2.scatter(res_ts['d'], res_ts['r'], facecolors='none', edgecolors='green', alpha=0.7)
            
            ax2.set_title(f"Teste: RUL vs Difference Feature\n(Truncamento Natural) | Corr: {corr_ts:.4f}")
            ax2.set_xlabel("Difference Feature")
            plots_design(ax2, fig, tamanho_figura=2)
            
            plt.tight_layout()
            plt.show()
            
if __name__ == "__main__":
    menu_visualizacao_tratamento()