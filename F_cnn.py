import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

# Importações dos seus módulos locais
from tratamento_03 import preparar_dados_redes_neurais 
from importacao_01 import plots_design

# =====================================================================
# 1. CONFIGURÁVEIS (Parâmetros Iniciais e Espaço de Busca)
# =====================================================================
tw = 30  
VAL_SPLIT = 0.2
MAX_TRIALS = 30  

# --- NOVOS CONFIGURÁVEIS ---
USAR_DIFFERENCE_FEATURE = True  # True = 2 features (Sinal Fundido + DF) | False = 1 feature (Apenas Sinal Fundido)
METRICA_LOSS = 'rmse'          # Opções: 'mse', 'rmse' ou 'phm08'

# Espaço de busca (Baseado na Tabela 5)
OPCOES_LAYERS = [1, 2, 3, 4]
OPCOES_EPOCHS = [20, 25, 30, 40, 50, 60, 80, 100, 110, 120]
OPCOES_BATCH = [64, 128, 256]
OPCOES_FILTERS = [32, 64, 128, 256]
OPCOES_UNITS = [32, 64, 128, 256]

# =====================================================================
# FUNÇÕES DE LOSS CUSTOMIZADAS
# =====================================================================
def rmse_loss(y_true, y_pred):
    """Calcula o Root Mean Squared Error (RMSE)."""
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - tf.cast(y_true, tf.float32))))

def phm08_loss(y_true, y_pred):
    """Calcula a penalidade assimétrica do PHM08 Data Challenge."""
    d = y_pred - tf.cast(y_true, tf.float32)
    # d < 0: Early prediction (divisor 13) | d >= 0: Late prediction (divisor 10)
    score = tf.where(d < 0, tf.math.exp(-d / 13.0) - 1.0, tf.math.exp(d / 10.0) - 1.0)
    # Usamos reduce_mean em vez da soma direta (N) para manter o gradiente estável no treinamento por batch
    return tf.reduce_mean(score) 

def selecionar_funcao_loss():
    if METRICA_LOSS.lower() == 'rmse': return rmse_loss
    if METRICA_LOSS.lower() == 'phm08': return phm08_loss
    return 'mse'

# =====================================================================
# 2. MODELO
# =====================================================================
def build_model(hp):
    model = tf.keras.models.Sequential()
    
    filters = hp.Choice('filters', OPCOES_FILTERS)
    lstm_units = hp.Choice('units', OPCOES_UNITS)
    num_layers = hp.Choice('num_layers', OPCOES_LAYERS)
    
    num_features = 2 if USAR_DIFFERENCE_FEATURE else 1
    
    # 1. Camada Convolucional (1D-CNN)
    model.add(tf.keras.layers.Conv1D(
        filters=filters, kernel_size=3, padding='same', 
        activation='relu', input_shape=(tw, num_features)
    ))
    # 2. Camada de Agrupamento (Max Pooling)
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
    
    # 3. Camada LSTM
    model.add(tf.keras.layers.LSTM(units=lstm_units))
    
    # 4. Camadas Densas (Fully Connected)
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(units=50, activation='relu'))
        
    # 5. Camada de Saída
    model.add(tf.keras.layers.Dense(units=1))
    
    otimizador_mac = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    
    # Aplica a função de perda dinâmica escolhida nos configuráveis
    model.compile(optimizer=otimizador_mac, loss=selecionar_funcao_loss())
    
    return model

class CustomCnnTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        kwargs['batch_size'] = hp.Choice('batch_size', OPCOES_BATCH)
        kwargs['epochs'] = hp.Choice('epochs', OPCOES_EPOCHS)
        return super().run_trial(trial, *args, **kwargs)

# =====================================================================
# 3. PLOTS
# =====================================================================
def plotar_efeito_unico(df, hp_name, x_label, filename):
    if hp_name not in df.columns: return
    
    efeito = df.groupby(hp_name)['score'].mean()
    fig, ax = plt.subplots()
    ax.plot(efeito.index.astype(str), efeito.values, marker='o', linestyle='-', color='black')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Desempenho Médio ({METRICA_LOSS.upper()})")
    
    plots_design(ax, fig=fig, tamanho_figura=2)
    
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

def gerar_graficos_tuner(tuner):
    os.makedirs('imagens', exist_ok=True)
    
    dados = []
    for t in tuner.oracle.trials.values():
        if t.score is not None:
            linha = t.hyperparameters.values.copy()
            linha['score'] = t.score 
            dados.append(linha)
            
    df_res = pd.DataFrame(dados)
    if df_res.empty: return
    
    plotar_efeito_unico(df_res, 'num_layers', 'Number of layers', 'imagens/hp_a_layers.png')
    plotar_efeito_unico(df_res, 'epochs', 'Number of epochs', 'imagens/hp_b_epochs.png')
    plotar_efeito_unico(df_res, 'filters', 'Filter size (First layer)', 'imagens/hp_c1_filters.png')
    plotar_efeito_unico(df_res, 'units', 'Number of units (First layer)', 'imagens/hp_c2_units.png')
    plotar_efeito_unico(df_res, 'batch_size', 'Batch size', 'imagens/hp_d_batch_size.png')

# =====================================================================
# 4. EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    # 4.1 Carregamento e Preparo (Passando a nova flag se a sua função tratar isso internamente)
    X_train, Y_train, X_test = preparar_dados_redes_neurais(seq_len=tw, usar_df=USAR_DIFFERENCE_FEATURE)
    
    # 4.2 Instanciação do Tuner Customizado
    tuner = CustomCnnTuner(
        hypermodel=build_model,
        objective='val_loss', 
        max_trials=MAX_TRIALS,
        directory='tuner_results',
        project_name='cnn_lstm_bayes_custom',
        overwrite=True
    )
    
    print(f"\n--- Iniciando Otimização Bayesiana com Loss: {METRICA_LOSS.upper()} ---")
    tuner.search(X_train, Y_train, validation_split=VAL_SPLIT, verbose=1)
    
    # 4.3 Salvar Hiperparâmetros em TXT
    best_hp = tuner.get_best_hyperparameters(1)[0]
    with open("melhores_hiperparametros.txt", "w", encoding='utf-8') as f:
        f.write(f"Melhores Hiperparametros Encontrados (Loss: {METRICA_LOSS.upper()}):\n")
        f.write(f"- Difference Feature: {'Ativado' if USAR_DIFFERENCE_FEATURE else 'Desativado'}\n")
        for param, value in best_hp.values.items():
            f.write(f"- {param}: {value}\n")
    print("\n[+] Configurações otimizadas salvas em 'melhores_hiperparametros.txt'.")
    
    # 4.4 Plotagem das análises de influência
    gerar_graficos_tuner(tuner)
    print("[+] Gráficos dos efeitos (tamanho 2) salvos na pasta 'imagens/'.")
    
    # 4.5 Treinamento final com os melhores hiperparâmetros
    print("\n--- Treinando o modelo final otimizado ---")
    melhor_modelo = tuner.hypermodel.build(best_hp)
    melhor_modelo.fit(
        X_train, Y_train, 
        epochs=best_hp.get('epochs'), 
        batch_size=best_hp.get('batch_size'), 
        verbose=1
    )
    
    # 4.6 Salvar modelo
    melhor_modelo.save('FD00x_cnn.keras')
    print("[+] Modelo salvo com sucesso como 'FD00x_cnn.keras'!")