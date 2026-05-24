import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

# Importações dos seus módulos locais
from C_tratamento import preparar_dados_redes_neurais 
from A_importacao import plots_design

# =====================================================================
# 1. CONFIGURÁVEIS (Parâmetros Iniciais e Espaço de Busca)
# =====================================================================
tw = 30  
VAL_SPLIT = 0.2
MAX_TRIALS = 40  

# --- CONFIGURAÇÕES DE DADOS E MÉTRICAS ---
USAR_DIFFERENCE_FEATURE = True  
METRICA_LOSS = 'rmse'           

# --- FLAGS DE OTIMIZAÇÃO (Ligar/Desligar Otimização Bayesiana por parâmetro) ---
OTIMIZAR_LR       = True
OTIMIZAR_FILTERS  = True
OTIMIZAR_UNITS    = True
OTIMIZAR_L2_REG   = True
OTIMIZAR_BATCH    = True
OTIMIZAR_LAYERS   = False  # Ex: mantendo fixo em 2 camadas
OTIMIZAR_EPOCHS   = False  # Com EarlyStopping, melhor deixar Fixo e alto
OTIMIZAR_MOMENTUM = False  # Adam já adapta bem o momentum padrão

# --- ESPAÇO DE BUSCA BAYESIANA (Tupla) vs VALOR PADRÃO (Número) ---
# Formato: (min_value, max_value, step/sampling) if OTIMIZAR else VALOR_FIXO

HP_LR        = (5e-4, 1e-2, 'log')   if OTIMIZAR_LR       else 0.001
HP_FILTERS   = (32, 256, 32)         if OTIMIZAR_FILTERS  else 64
HP_UNITS     = (32, 256, 32)         if OTIMIZAR_UNITS    else 64
HP_L2_REG    = (1e-10, 1e-2, 'log')  if OTIMIZAR_L2_REG   else 1e-5
HP_BATCH     = (100, 500, 100)       if OTIMIZAR_BATCH    else 128
HP_LAYERS    = (1, 3, 1)             if OTIMIZAR_LAYERS   else 2
HP_EPOCHS    = (50, 200, 10)         if OTIMIZAR_EPOCHS   else 150 
HP_MOMENTUM  = (0.3, 0.95, 'linear') if OTIMIZAR_MOMENTUM else 0.9

# =====================================================================
# FUNÇÕES DE LOSS CUSTOMIZADAS
# =====================================================================
def rmse_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - tf.cast(y_true, tf.float32))))

def phm08_loss(y_true, y_pred):
    d = y_pred - tf.cast(y_true, tf.float32)
    score = tf.where(d < 0, tf.math.exp(-d / 13.0) - 1.0, tf.math.exp(d / 10.0) - 1.0)
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
    
    # Processa os hiperparâmetros (Avalia se é tupla para otimizar ou valor fixo)
    filters    = hp.Int('filters', min_value=HP_FILTERS[0], max_value=HP_FILTERS[1], step=HP_FILTERS[2]) if OTIMIZAR_FILTERS else HP_FILTERS
    lstm_units = hp.Int('units', min_value=HP_UNITS[0], max_value=HP_UNITS[1], step=HP_UNITS[2]) if OTIMIZAR_UNITS else HP_UNITS
    num_layers = hp.Int('num_layers', min_value=HP_LAYERS[0], max_value=HP_LAYERS[1], step=HP_LAYERS[2]) if OTIMIZAR_LAYERS else HP_LAYERS
    
    lr         = hp.Float('learning_rate', min_value=HP_LR[0], max_value=HP_LR[1], sampling=HP_LR[2]) if OTIMIZAR_LR else HP_LR
    momentum   = hp.Float('momentum', min_value=HP_MOMENTUM[0], max_value=HP_MOMENTUM[1]) if OTIMIZAR_MOMENTUM else HP_MOMENTUM
    l2_reg     = hp.Float('l2_reg', min_value=HP_L2_REG[0], max_value=HP_L2_REG[1], sampling=HP_L2_REG[2]) if OTIMIZAR_L2_REG else HP_L2_REG
    
    regularizer = tf.keras.regularizers.l2(l2_reg)
    num_features = 2 if USAR_DIFFERENCE_FEATURE else 1
    
    # 1. Camada Convolucional (1D-CNN) com L2
    model.add(tf.keras.layers.Conv1D(
        filters=filters, kernel_size=3, padding='same', 
        activation='relu', input_shape=(tw, num_features),
        kernel_regularizer=regularizer
    ))
    # 2. Max Pooling
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
    
    # 3. LSTM com L2
    model.add(tf.keras.layers.LSTM(units=lstm_units, kernel_regularizer=regularizer))
    
    # 4. Camadas Densas com L2
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer=regularizer))
        
    # 5. Saída
    model.add(tf.keras.layers.Dense(units=1))
    
    # 6. Otimizador
    otimizador_mac = tf.keras.optimizers.legacy.Adam(learning_rate=lr, beta_1=momentum)
    model.compile(optimizer=otimizador_mac, loss=selecionar_funcao_loss())
    
    return model

class CustomCnnTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        kwargs['batch_size'] = hp.Int('batch_size', min_value=HP_BATCH[0], max_value=HP_BATCH[1], step=HP_BATCH[2]) if OTIMIZAR_BATCH else HP_BATCH
        kwargs['epochs'] = hp.Int('epochs', min_value=HP_EPOCHS[0], max_value=HP_EPOCHS[1], step=HP_EPOCHS[2]) if OTIMIZAR_EPOCHS else HP_EPOCHS
        return super().run_trial(trial, *args, **kwargs)

# =====================================================================
# 3. PLOTS
# =====================================================================
def plotar_efeito_unico(df, hp_name, x_label, filename):
    if hp_name not in df.columns: return # Se a flag for False, a coluna não existe e o gráfico é ignorado
    
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
    
    plotar_efeito_unico(df_res, 'num_layers', 'Network depth', 'imagens/hp_a_layers.png')
    plotar_efeito_unico(df_res, 'epochs', 'Number of epochs', 'imagens/hp_b_epochs.png')
    plotar_efeito_unico(df_res, 'filters', 'Filter size (CNN)', 'imagens/hp_c1_filters.png')
    plotar_efeito_unico(df_res, 'units', 'Number of units (LSTM)', 'imagens/hp_c2_units.png')
    plotar_efeito_unico(df_res, 'batch_size', 'Batch size', 'imagens/hp_d_batch_size.png')
    plotar_efeito_unico(df_res, 'learning_rate', 'Initial learning rate', 'imagens/hp_e_lr.png')
    plotar_efeito_unico(df_res, 'momentum', 'Momentum', 'imagens/hp_f_momentum.png')
    plotar_efeito_unico(df_res, 'l2_reg', 'L2 Regularization', 'imagens/hp_g_l2.png')

# =====================================================================
# 4. EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    X_train, Y_train, X_test = preparar_dados_redes_neurais(seq_len=tw, usar_df=USAR_DIFFERENCE_FEATURE)
    
    tuner = CustomCnnTuner(
        hypermodel=build_model,
        objective='val_loss', 
        max_trials=MAX_TRIALS,
        directory='tuner_results',
        project_name='cnn_lstm_bayes_custom',
        overwrite=True
    )
    
    # Callback de EarlyStopping: Para quando 'val_loss' não melhora por 10 épocas, restaurando os melhores pesos
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"\n--- Iniciando Otimização Bayesiana com Loss: {METRICA_LOSS.upper()} ---")
    tuner.search(X_train, Y_train, validation_split=VAL_SPLIT, verbose=1, callbacks=[early_stop])
    
    best_hp = tuner.get_best_hyperparameters(1)[0]
    
    with open("melhores_hiperparametros.txt", "w", encoding='utf-8') as f:
        f.write(f"Melhores Hiperparametros Encontrados (Loss: {METRICA_LOSS.upper()}):\n")
        f.write(f"- Difference Feature: {'Ativado' if USAR_DIFFERENCE_FEATURE else 'Desativado'}\n")
        for param, value in best_hp.values.items():
            if isinstance(value, float) and value < 0.01: f.write(f"- {param}: {value:.2e}\n")
            else: f.write(f"- {param}: {value}\n")
                
    print("\n[+] Configurações otimizadas salvas em 'melhores_hiperparametros.txt'.")
    gerar_graficos_tuner(tuner)
    print("[+] Gráficos dos efeitos salvos na pasta 'imagens/'.")
    
    print("\n--- Treinando o modelo final otimizado ---")
    melhor_modelo = tuner.hypermodel.build(best_hp)
    
    # Resgata o valor final do batch_size e epochs
    final_epochs = best_hp.get('epochs') if OTIMIZAR_EPOCHS else HP_EPOCHS
    final_batch = best_hp.get('batch_size') if OTIMIZAR_BATCH else HP_BATCH
    
    # Treinamento final também usando EarlyStopping
    melhor_modelo.fit(
        X_train, Y_train, 
        validation_split=VAL_SPLIT, 
        epochs=final_epochs, 
        batch_size=final_batch, 
        callbacks=[early_stop],
        verbose=1
    )
    
    melhor_modelo.save('FD00x_cnn.keras')
    print("[+] Modelo salvo com sucesso como 'FD00x_cnn.keras'!")