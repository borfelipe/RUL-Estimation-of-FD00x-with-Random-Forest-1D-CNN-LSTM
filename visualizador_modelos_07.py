import os
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import tensorflow as tf
import matplotlib.image as mpimg
import visualkeras
from tensorflow.keras.models import load_model


def visualizar_random_forest():
    print("\n--- Visualização da Random Forest ---")
    nome_arquivo = 'FD001_rf.joblib' # Nome padrão baseado no seu projeto
    
    if not os.path.exists(nome_arquivo):
        nome_arquivo = input(f"Arquivo '{nome_arquivo}' não encontrado. Digite o nome correto do arquivo (ex: modelo_rf.joblib): ").strip()
        if not os.path.exists(nome_arquivo):
            print("Arquivo não encontrado. Retornando ao menu principal.")
            return

    try:
        print(f"Carregando modelo '{nome_arquivo}'...")
        rf_model = joblib.load(nome_arquivo)
        
        # Pega a quantidade de árvores na floresta
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
        
        # Puxando as features que você usou no treinamento (FS e DF)
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
        print(f"Erro ao carregar ou visualizar a Random Forest: {e}")


def visualizar_cnn():
    print("\n--- Visualização da CNN ---")
    nome_arquivo = 'FD001_cnn.keras' # Nome padrão baseado no seu projeto
    
    if not os.path.exists(nome_arquivo):
        nome_arquivo = input(f"Arquivo '{nome_arquivo}' não encontrado. Digite o nome correto do arquivo (ex: modelo_cnn.keras): ").strip()
        if not os.path.exists(nome_arquivo):
            print("Arquivo não encontrado. Retornando ao menu principal.")
            return

    nome_imagem = input("Qual nome deseja dar à figura da arquitetura gerada? (Pressione Enter para usar 'arquitetura_cnn.png'): ").strip()
    if not nome_imagem:
        nome_imagem = 'arquitetura_cnn.png'
    if not nome_imagem.endswith('.png'):
        nome_imagem += '.png'

    try:
        print(f"Carregando modelo '{nome_arquivo}'...")
        # compile=False pois só queremos a arquitetura, não vamos treinar
        cnn_model = tf.keras.models.load_model(nome_arquivo, compile=False) 
        
        print(f"Gerando diagrama e salvando em '{nome_imagem}'...")
        tf.keras.utils.plot_model(cnn_model, 
                                  to_file=nome_imagem, 
                                  show_shapes=True, 
                                  show_layer_names=True, 
                                  rankdir='TB',
                                  dpi=200)
        
        print(f"Sucesso! Imagem salva como '{nome_imagem}'.")
        
        modelo = load_model('FD001_cnn.keras', compile=False)
        visualkeras.layered_view(modelo, legend=True, to_file='arquitetura_3d.png').show()
        try:
            img = mpimg.imread(nome_imagem)
            plt.figure(figsize=(10, 14))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Arquitetura da CNN", fontsize=16)
            plt.show()
        except:
            pass # Se por acaso falhar a exibição, a imagem já está salva na pasta
            
    except ImportError as e:
        print(f"Erro de dependência: Certifique-se de ter o 'pydot' e o 'graphviz' instalados no seu sistema.\nDetalhes: {e}")
    except Exception as e:
        print(f"Erro ao carregar ou visualizar a CNN: {e}")


def menu_principal():
    while True:
        print("\n" + "="*40)
        print("  DASHBOARD: VISUALIZAÇÃO DOS MODELOS")
        print("="*40)
        print("1 - Visualizar árvore da Random Forest")
        print("2 - Visualizar arquitetura da CNN")
        print("0 - Sair")
        print("="*40)
        
        opcao = input("Escolha uma opção: ").strip()
        
        if opcao == '1':
            visualizar_random_forest()
        elif opcao == '2':
            visualizar_cnn()
        elif opcao == '0':
            print("Encerrando visualizador. Bons estudos no TCC/Projeto Final!")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    menu_principal()