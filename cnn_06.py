import tensorflow as tf
import keras_tuner as kt
from tratamento_03 import preparar_dados_redes_neurais

tw = 30 # time window lenght

# ==============================================================================
# Otimização Bayesiana - Ref: Li e He, 2020
# ==============================================================================
class CNNHyperModel(kt.HyperModel):
    """
    Construtor do modelo CNN seguindo a Tabela 2 do artigo Li e He (2020).
    """
    def build(self, hp):
        model = tf.keras.models.Sequential()
        
        # L2 Regularization range: [1e-10, 1e-2] (Li e He, 2020)
        l2_reg = hp.Float('l2_reg', min_value=1e-10, max_value=1e-2, sampling='log')
        
        # Camada Conv1D
        model.add(tf.keras.layers.Conv1D(
            filters=32, 
            kernel_size=3, 
            padding='same', 
            activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            input_shape=(tw, 2)
        ))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
        
        # Camada LSTM
        model.add(tf.keras.layers.LSTM(
            units=16, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ))

        # Network Depth range: [1, 3] (Li e He, 2020)
        # Otimiza dinamicamente o número de camadas Densas ocultas
        network_depth = hp.Int('network_depth', min_value=1, max_value=3)
        for i in range(network_depth):
            model.add(tf.keras.layers.Dense(
                units=50, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            ))
            
        # Camada de Saída
        model.add(tf.keras.layers.Dense(units=1))

        # Initial learning rate range: [5e-4, 1e-2] (Li e He, 2020)
        lr = hp.Float('learning_rate', min_value=5e-4, max_value=1e-2, sampling='log')
        otimizador = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        
        model.compile(optimizer=otimizador, loss='mse')
        return model

    def fit(self, hp, model, *args, **kwargs):
        # Batchsize range: [200, 1000] (Li e He, 2020)
        # Utilizando a função fit customizada para o Tuner otimizar também o tamanho do lote
        batch_size = hp.Int('batch_size', min_value=200, max_value=1000, step=100)
        return model.fit(*args, batch_size=batch_size, **kwargs)

def otimizar_cnn():
    print("\n--- Iniciando Otimização Bayesiana da CNN ---")
    X_train, Y_train, X_test = preparar_dados_redes_neurais(seq_len=tw)
    
    tuner = kt.BayesianOptimization(
        hypermodel=CNNHyperModel(),
        objective='val_loss',
        max_trials=20,          # Número de iterações do Processo Gaussiano
        num_initial_points=5,   # Pontos iniciais para exploração global
        directory='otimizacao_dir',
        project_name='cnn_bayesian_optimization',
        overwrite=True
    )
    
    # Early Stopping para não prolongar o treinamento de configurações ruins
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    # Realiza a busca bayesiana
    tuner.search(X_train, Y_train, epochs=30, validation_split=0.2, callbacks=[stop_early], verbose=1)
    
    melhores_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nMelhores hiperparâmetros encontrados para a CNN (Li e He, 2020):")
    print(f"- Network Depth (Camadas Densas): {melhores_hps.get('network_depth')}")
    print(f"- Learning Rate: {melhores_hps.get('learning_rate'):.5f}")
    print(f"- Batch Size: {melhores_hps.get('batch_size')}")
    print(f"- L2 Regularization: {melhores_hps.get('l2_reg'):.5e}")
    
    # Constrói e treina o melhor modelo final
    melhor_modelo_cnn = tuner.hypermodel.build(melhores_hps)
    melhor_modelo_cnn.fit(
        X_train, Y_train, 
        epochs=30, 
        batch_size=melhores_hps.get('batch_size'), 
        validation_split=0.2,
        verbose=1
    )
    
    melhor_modelo_cnn.save('FD00x_cnn.keras')
    print("Modelo CNN 1D-LSTM otimizado salvo com sucesso!")
    return melhores_hps

# Execução do script
if __name__ == "__main__":
    params_cnn = otimizar_cnn()
    
"""
import numpy as np
import tensorflow as tf
from tratamento_03 import preparar_dados_redes_neurais 

# Prepara os dados de treinamento extraindo o sinal fundido e o feature de diferença
X_train, Y_train, X_test = preparar_dados_redes_neurais(seq_len=30) # Ensarioğlu Sec. 3.1.4: Tamanho da janela de tempo (tw)

# Inicialização do modelo híbrido 1D-CNN-LSTM
model = model = tf.keras.models.Sequential()

# 1. Camada Convolucional (1D-CNN) para extrair features espaciais
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(30, 2)))  # Ensarioğlu Tab. 6 / Sec 3.1.8/3.1.9
# 2. Camada de Agrupamento (Max Pooling) para reduzir complexidade e overfitting
model.add(tf.keras.layers.MaxPooling1D(pool_size=3))  # Ensarioğlu Sec. 3.1.8

# 3. Camada LSTM para revelar informações temporais
model.add(tf.keras.layers.LSTM(units=16))  # Ensarioğlu Sec. 3.1.8

# 4. Primeira Camada Densa (Fully Connected) para suavizar a matriz de features
model.add(tf.keras.layers.Dense(units=50, activation='relu')) # Ensarioğlu Sec. 3.1.8

# 5. Segunda Camada Densa (Fully Connected)
model.add(tf.keras.layers.Dense(units=50, activation='relu'))  # Ensarioğlu Sec. 3.1.8:

# 6. Camada de Saída
model.add(tf.keras.layers.Dense(units=1))  # Ensarioğlu Sec. 2.6: Camada de saída com exatamente 1 neurônio para a predição final do RUL

# Compilação do modelo com os parâmetros de otimização e perda
otimizador_mac = tf.keras.optimizers.legacy.Adam(learning_rate=0.001) # Ensarioğlu Sec. 2.6 / 3.1
model.compile(optimizer=otimizador_mac, loss='mse') 

# Treinamento do modelo
history = model.fit(X_train, Y_train, epochs=30, batch_size=256, verbose=1) # Ensarioğlu Tab. 6
model.save('modelo_1d_cnn_lstm.keras')
"""