import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pwlf
from C_tratamento import *
from A_importacao import *

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
            modo_saida = input("Visualizar em (g)rade inline ou (s)alvar individualmente? ").strip().lower()
            
            d_norm = estagios["normalizado"][estagios["normalizado"]['unidade'] == unid]
            d_filt = estagios["filtrado"][estagios["filtrado"]['unidade'] == unid]
            
            # Mapeamento para salvar o nome exato do sensor
            mapa_sensores = {
                'T24': 2, 'T30': 3, 'T50': 4, 'P30': 7, 'Nf': 8, 'Nc': 9, 
                'PS30': 11, 'phi': 12, 'NRf': 13, 'NRc': 14, 'BPR': 15, 
                'htBleed': 17, 'W31': 20, 'W32': 21
            }
            
            if modo_saida == 'g':
                fig, axs = plt.subplots(7, 2, figsize=(14, 18), sharex=True)
                axs = axs.flatten()
                for i, col in enumerate(COLUNAS_MANTER):
                    axs[i].set_xlabel("Ciclo")
                    axs[i].set_ylabel(f"Leitura normalizada de {col}")
                    axs[i].plot(d_norm['ciclo'], d_norm[col], color='gray', alpha=0.5, label='Original')
                    axs[i].plot(d_filt['ciclo'], d_filt[col], color='blue', label='Filtrado')
                    axs[i].set_title(col)
                    plots_design(axs[i], tamanho_figura=3, posicao_legenda='fora') 
                
                plots_design(axs[0], fig=fig, tamanho_figura=3, is_grade=True)
                plt.show()
                
            else:
                os.makedirs('imagens', exist_ok=True)
                for col in COLUNAS_MANTER:
                    num_sensor = mapa_sensores.get(col, 0)
                    fig, ax = plt.subplots()
                    ax.set_xlabel("Ciclo")
                    ax.set_ylabel(f"Leitura normalizada de {col}")
                    ax.plot(d_norm['ciclo'], d_norm[col], color='gray', alpha=0.5, label='Original')
                    ax.plot(d_filt['ciclo'], d_filt[col], color='blue', label='Filtrado')
                    ax.set_title(col)
                    
                    plots_design(ax, fig=fig, tamanho_figura=3, posicao_legenda='fora')
                    plt.savefig(f"imagens/FD00x-S{num_sensor}-NF.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
                    plt.close(fig)
                print("[+] Imagens individuais (tamanho 3) guardadas em 'imagens/'.")

        elif op == '2':
            modo_saida = input("Visualizar em (g)rade inline ou (s)alvar individualmente? ").strip().lower()
            
            df_train = estagios["fs_train"] 
            df_test = estagios["fs_test"]
            
            if modo_saida == 'g':
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                
                for _, d in df_train.groupby('unidade'):
                    axs[0].plot(d['ciclo'], d['FS'])
                axs[0].set_xlabel("Ciclo")
                axs[0].set_ylabel("Sinal Fundido")
                axs[0].set_ylim(-0.05, 1.05) 
                axs[0].set_title("Treino")
                plots_design(axs[0], tamanho_figura=2, posicao_legenda='dentro') 
                
                for _, d in df_test.groupby('unidade'):
                    axs[1].plot(d['ciclo'], d['FS'])
                axs[1].set_xlabel("Ciclo")
                axs[1].set_ylabel("Sinal Fundido")
                axs[1].set_ylim(-0.05, 1.05) 
                axs[1].set_title("Teste")
                plots_design(axs[1], tamanho_figura=2, posicao_legenda='dentro')
                
                plots_design(axs[0], fig=fig, tamanho_figura=2, is_grade=True)
                plt.show()
                
            else:
                os.makedirs('imagens', exist_ok=True)
                
                fig1, ax1 = plt.subplots()
                for _, d in df_train.groupby('unidade'):
                    ax1.plot(d['ciclo'], d['FS'])
                ax1.set_xlabel("Ciclo")
                ax1.set_ylabel("Sinal Fundido")
                ax1.set_ylim(-0.05, 1.05) 
                plots_design(ax1, fig=fig1, tamanho_figura=2, posicao_legenda='dentro')
                plt.savefig("imagens/FD00x-FS-Treino.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
                plt.close(fig1)

                fig2, ax2 = plt.subplots()
                for _, d in df_test.groupby('unidade'):
                    ax2.plot(d['ciclo'], d['FS'])
                ax2.set_xlabel("Ciclo")
                ax2.set_ylabel("Sinal Fundido")
                ax2.set_ylim(-0.05, 1.05) 
                plots_design(ax2, fig=fig2, tamanho_figura=2, posicao_legenda='dentro')
                plt.savefig("imagens/FD00x-FS-Teste.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
                plt.close(fig2)
                
                print("[+] Imagens individuais (tamanho 2) guardadas em 'imagens/'.")
        
        elif op == '3':
            n_seg = int(input("Número de segmentos (ex: 2 para 1 Change-Point): "))
            modo = input("Deseja visualizar todos os motores (T) ou um específico (E)? ").strip().lower()
            modo_saida = input("Visualizar em (g)rade inline ou (s)alvar individualmente? ").strip().lower()
            
            fs_data = estagios["fs_train"]
            
            if modo == 't':
                print(f"Processando PWLF para todos os motores com {n_seg} segmentos... Aguarde.")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for unit, dados_m in fs_data.groupby('unidade'):
                    my_pwlf = pwlf.PiecewiseLinFit(dados_m['ciclo'].values, dados_m['FS'].values)
                    cps_m = my_pwlf.fit(n_seg)
                    cp_principal = cps_m[1]
                    
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
                
                plots_design(ax, fig=fig, tamanho_figura=1)
                
                if modo_saida == 'g':
                    plt.show()
                else:
                    os.makedirs('imagens', exist_ok=True)
                    plt.savefig("imagens/FD00x-CPD-Todos.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
                    plt.close(fig)
                    print("[+] Imagem salva em 'imagens/'.")
                
            else:
                unid = int(input("Número do motor: "))
                dados = fs_data[fs_data['unidade'] == unid].copy()
                
                if dados.empty:
                    print(f"Motor {unid} não encontrado.")
                else:
                    my_pwlf = pwlf.PiecewiseLinFit(dados['ciclo'].values, dados['FS'].values)
                    cps_encontrados = my_pwlf.fit(n_seg)
                    cp_principal = cps_encontrados[1] 
                    
                    max_ciclo = dados['ciclo'].max()
                    dados['RUL_CPD_VIS'] = np.where(
                        dados['ciclo'] < cp_principal, 
                        max_ciclo - cp_principal, 
                        dados['RUL']
                    )
                    
                    x_hat = np.linspace(dados['ciclo'].min(), dados['ciclo'].max(), 100)
                    y_hat = my_pwlf.predict(x_hat)
                    
                    if modo_saida == 'g':
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        ax1.plot(dados['ciclo'], dados['FS'], 'k-', alpha=0.5, label='Fused Signal')
                        ax1.plot(x_hat, y_hat, 'b-', linewidth=2, label=f'PWLF ({n_seg} segs)')
                        for cp in cps_encontrados[1:-1]:
                            ax1.axvline(cp, color='r', linestyle='--', alpha=0.7, label=f'CP: {int(cp)}')
                        ax1.set_xlabel('Ciclo')
                        ax1.set_ylabel('Sinal Fundido')
                        ax1.set_title('Detecção de Change-Point (PWLF)')
                        ax1.legend()
                        
                        ax2.plot(dados['ciclo'], dados['RUL'], 'k--', label='RUL Linear')
                        ax2.plot(dados['ciclo'], dados['RUL_CPD_VIS'], 'b', label=f'RUL CPD')
                        ax2.axvline(cp_principal, color='r', linestyle=':', label='Início Degradação')
                        ax2.set_xlabel('Ciclo')
                        ax2.set_ylabel('RUL')
                        ax2.set_title('Ajuste dos Rótulos RUL')
                        ax2.legend()
                        
                        plots_design(ax1, tamanho_figura=2, posicao_legenda='dentro')
                        plots_design(ax2, tamanho_figura=2, posicao_legenda='dentro')
                        plots_design(ax1, fig=fig, tamanho_figura=2, is_grade=True)
                        plt.show()
                        
                    else:
                        os.makedirs('imagens', exist_ok=True)
                        
                        fig1, ax1 = plt.subplots()
                        ax1.plot(dados['ciclo'], dados['FS'], 'k-', alpha=0.5, label='Fused Signal')
                        ax1.plot(x_hat, y_hat, 'b-', linewidth=2, label=f'PWLF ({n_seg} segs)')
                        for cp in cps_encontrados[1:-1]:
                            ax1.axvline(cp, color='r', linestyle='--', alpha=0.7, label=f'CP: {int(cp)}')
                        ax1.set_xlabel('Ciclo')
                        ax1.set_ylabel('Sinal Fundido')
                        ax1.set_title('Detecção de Change-Point (PWLF)')
                        ax1.legend()
                        plots_design(ax1, fig=fig1, tamanho_figura=2, posicao_legenda='dentro')
                        plt.savefig(f"imagens/FD00x-M{unid}-CPD.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
                        plt.close(fig1)

                        fig2, ax2 = plt.subplots()
                        ax2.plot(dados['ciclo'], dados['RUL'], 'k--', label='RUL Linear')
                        ax2.plot(dados['ciclo'], dados['RUL_CPD_VIS'], 'b', label=f'RUL CPD')
                        ax2.axvline(cp_principal, color='r', linestyle=':', label='Início Degradação')
                        ax2.set_xlabel('Ciclo')
                        ax2.set_ylabel('RUL')
                        ax2.set_title('Ajuste dos Rótulos RUL')
                        ax2.legend()
                        plots_design(ax2, fig=fig2, tamanho_figura=2, posicao_legenda='dentro')
                        plt.savefig(f"imagens/FD00x-M{unid}-Rotulo.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
                        plt.close(fig2)
                        
                        print("[+] Imagens individuais (tamanho 2) guardadas em 'imagens/'.")

        elif op == '4':
            modo_saida = input("Visualizar em (g)rade inline ou (s)alvar individualmente? ").strip().lower()
            
            res_tr = estagios["corr_data_train"]
            corr_tr = estagios["corr_val_train"]
            
            _, _, y_test_true = importar_dados() 
            df_test = estagios["fs_test"]
            
            res_test_list = []
            for i, (unit, data) in enumerate(df_test.groupby('unidade')):
                ultimo_ponto = data.iloc[-1]
                res_test_list.append({
                    'u': unit, 
                    'd': ultimo_ponto['DF'], 
                    'r': y_test_true[i] 
                })
            
            res_ts = pd.DataFrame(res_test_list)
            corr_ts = res_ts['d'].corr(res_ts['r'])
            
            if modo_saida == 'g':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
                
                ax1.scatter(res_tr['d'], res_tr['r'], facecolors='none', edgecolors='blue', alpha=0.7)
                ax1.set_title(f"Treino: RUL vs Difference Feature\n(Truncamento Aleatório) | Corr: {corr_tr:.4f}")
                ax1.set_xlabel("Difference Feature ")
                ax1.set_ylabel("RUL")
                plots_design(ax1, tamanho_figura=2)
                
                ax2.scatter(res_ts['d'], res_ts['r'], facecolors='none', edgecolors='green', alpha=0.7)
                ax2.set_title(f"Teste: RUL vs Difference Feature\n(Truncamento Natural) | Corr: {corr_ts:.4f}")
                ax2.set_xlabel("Difference Feature")
                plots_design(ax2, tamanho_figura=2)
                
                plots_design(ax1, fig=fig, tamanho_figura=2, is_grade=True)
                plt.tight_layout()
                plt.show()
                
            else:
                os.makedirs('imagens', exist_ok=True)
                
                fig1, ax1 = plt.subplots()
                ax1.scatter(res_tr['d'], res_tr['r'], facecolors='none', edgecolors='blue', alpha=0.7)
                ax1.set_title(f"Treino: RUL vs Difference Feature\n(Truncamento Aleatório) | Corr: {corr_tr:.4f}")
                ax1.set_xlabel("Difference Feature ")
                ax1.set_ylabel("RUL")
                plots_design(ax1, fig=fig1, tamanho_figura=2)
                plt.savefig("imagens/FD00x-DF-Treino.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
                plt.close(fig1)

                fig2, ax2 = plt.subplots()
                ax2.scatter(res_ts['d'], res_ts['r'], facecolors='none', edgecolors='green', alpha=0.7)
                ax2.set_title(f"Teste: RUL vs Difference Feature\n(Truncamento Natural) | Corr: {corr_ts:.4f}")
                ax2.set_xlabel("Difference Feature")
                ax2.set_ylabel("RUL")
                plots_design(ax2, fig=fig2, tamanho_figura=2)
                plt.savefig("imagens/FD00x-DF-Teste.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
                plt.close(fig2)
                
                print("[+] Imagens individuais (tamanho 2) guardadas em 'imagens/'.")
            
if __name__ == "__main__":
    menu_visualizacao_tratamento()