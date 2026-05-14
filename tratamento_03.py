import numpy as np
import pandas as pd
import pwlf
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from importacao_01 import importar_dados

COLUNAS_MANTER = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'PS30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']

# --- funções para tratamento

def filtro_passa_baixa(serie, cutoff=0.08, order=1):
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, serie)

def definir_condicoes_operacao(df_train, df_test):
    colunas_settings = ['Altitude', 'Mach', 'TRA']
    kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
    kmeans.fit(df_train[colunas_settings])
    df_train['cluster_op'] = kmeans.predict(df_train[colunas_settings])
    df_test['cluster_op'] = kmeans.predict(df_test[colunas_settings])
    
    return df_train, df_test

def normalizar_dados(df_train, df_test, no_ops=6, coluna_condicao='condicao_operacao'):
    if no_ops == 1:
        # Lógica original (Global MinMax)
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_train[COLUNAS_MANTER] = scaler.fit_transform(df_train[COLUNAS_MANTER])
        df_test[COLUNAS_MANTER] = scaler.transform(df_test[COLUNAS_MANTER].copy())
        
    else:
        # Nova lógica: Normalização Min-Max por condição de operação (m)
        # Identifica todas as condições de operação únicas no dataset de treino
        condicoes_operacao = df_train[coluna_condicao].unique()
        
        # Dicionário para armazenar os scalers ajustados no treino, 
        # garantindo que o teste use os mesmos min/max do treino para cada condição.
        scalers = {}
        
        for cond in condicoes_operacao:
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Máscaras booleanas para filtrar as linhas correspondentes à condição "m"
            mask_train = df_train[coluna_condicao] == cond
            mask_test = df_test[coluna_condicao] == cond
            
            # Ajusta (fit) e transforma no dataset de treino
            if mask_train.any():
                df_train.loc[mask_train, COLUNAS_MANTER] = scaler.fit_transform(
                    df_train.loc[mask_train, COLUNAS_MANTER]
                )
                scalers[cond] = scaler # Salva o scaler para usar no teste
            
            # Transforma no dataset de teste
            if mask_test.any() and cond in scalers:
                df_test.loc[mask_test, COLUNAS_MANTER] = scalers[cond].transform(
                    df_test.loc[mask_test, COLUNAS_MANTER]
                )

    return df_train, df_test

def filtrar_dados(df_train, df_test):
    for col in COLUNAS_MANTER:
        df_train[col] = df_train.groupby('unidade')[col].transform(lambda x: filtro_passa_baixa(x))
        df_test[col] = df_test.groupby('unidade')[col].transform(lambda x: filtro_passa_baixa(x) if len(x) > 3 else x)
    return df_train, df_test

def extrair_fused_signal(df_train, df_test):
    # 1. Arquitetura da Rede (conforme artigo)
    entrada = Input(shape=(14,))
    encoder_hidden = Dense(7, activation='relu')(entrada)
    sinal_fundido = Dense(1, activation='linear', name='fused')(encoder_hidden)
    decoder_hidden = Dense(7, activation='relu')(sinal_fundido)
    saida = Dense(14, activation='linear')(decoder_hidden)
    
    ae = Model(inputs=entrada, outputs=saida)
    extrator_fs = Model(inputs=entrada, outputs=sinal_fundido)
    
    ae.compile(optimizer='adam', loss='mse')
    
    # Treinamento com 80/20 split
    ae.fit(
        df_train[COLUNAS_MANTER], df_train[COLUNAS_MANTER], 
        epochs=30, batch_size=128, validation_split=0.2, verbose=0
    )
    
    # 2. Extração Bruta
    fs_train_bruto = extrator_fs.predict(df_train[COLUNAS_MANTER], verbose=0)
    fs_test_bruto = extrator_fs.predict(df_test[COLUNAS_MANTER], verbose=0)
    
    # 3. Correção de Direcionalidade
    # Verifica a correlação com o tempo (ciclo). Se for negativa (sinal caindo), invertemos.
    correlacao = np.corrcoef(df_train['ciclo'], fs_train_bruto.flatten())[0, 1]
    if correlacao < 0:
        fs_train_bruto = -fs_train_bruto
        fs_test_bruto = -fs_test_bruto
        
    # 4. Normalização do Fused Signal (0 a 1)
    scaler_fs = MinMaxScaler(feature_range=(0, 1))
    df_train['FS'] = scaler_fs.fit_transform(fs_train_bruto)
    df_test['FS'] = scaler_fs.transform(fs_test_bruto) # Importante: apenas transform no teste
    
    return df_train, df_test

def construir_difference_feature(fs_train, fs_test):
    fs_train['DF'] = (fs_train['FS'] - fs_train.groupby('unidade')['FS'].transform('first')).abs()
    fs_test['DF'] = (fs_test['FS'] - fs_test.groupby('unidade')['FS'].transform('first')).abs()
    
    # --- Dados de Correlação: Apenas Treino (Ciclos Aleatórios) ---
    res_train = []
    np.random.seed(42)
    for unit, data in fs_train.groupby('unidade'):
        idx = np.random.randint(0, len(data))
        pt = data.iloc[idx]
        res_train.append({'u': unit, 'd': pt['DF'], 'r': pt['RUL']})
        
    df_corr_train = pd.DataFrame(res_train)
    corr_train = df_corr_train['d'].corr(df_corr_train['r'])
    
    return fs_train, fs_test, df_corr_train, corr_train

def rotular_rul_cpd(fs_train):
    df_rotulado = fs_train.copy()
    df_rotulado['RUL_CPD'] = df_rotulado['RUL']
    cps = {}
    
    for unid, dados in df_rotulado.groupby('unidade'):
        my_pwlf = pwlf.PiecewiseLinFit(dados['ciclo'].values, dados['FS'].values)
        cp = my_pwlf.fit(2)[1]
        cps[unid] = cp
        df_rotulado.loc[(df_rotulado['unidade'] == unid) & (df_rotulado['ciclo'] < cp), 'RUL_CPD'] = dados['ciclo'].max() - cp # O RUL permanece constante até o CP, após o CP ele decai linearmente
        
    return df_rotulado, cps

def extrair_janelas_3d(df_train, df_test, seq_len):
    X_train, Y_train, X_test = [], [], []
    
    # Treino: Janelas deslizantes em cada motor
    for _, d in df_train.groupby('unidade'):
        v = d[['FS', 'DF']].values
        r = d['RUL_CPD'].values
        for i in range(len(v) - seq_len + 1):
            X_train.append(v[i:i+seq_len])
            Y_train.append(r[i+seq_len-1])
            
    # Teste: Apenas a última janela de cada motor (com padding se necessário)
    for _, d in df_test.groupby('unidade'):
        v = d[['FS', 'DF']].values
        if len(v) < seq_len:
            v = np.vstack([np.zeros((seq_len - len(v), 2)), v])
        X_test.append(v[-seq_len:])

    return np.array(X_train), np.array(Y_train), np.array(X_test)

# --- outputs

def obter_dados_tratados_2d():
    """Gera estágios de processamento para visualização e Random Forest."""
    df_bruto_train, df_bruto_test, _ = importar_dados()
    
    # Processamento sequencial
    df_t0, df_ts0 = definir_condicoes_operacao(df_bruto_train, df_bruto_test)
    df_t1, df_ts1 = normalizar_dados(df_t0, df_ts0, no_ops=6, coluna_condicao='cluster_op')
    df_t2, df_ts2 = filtrar_dados(df_t1.copy(), df_ts1.copy())
    
    fs_train, fs_test = extrair_fused_signal(df_t2, df_ts2)
    fs_train_rotulado, cps = rotular_rul_cpd(fs_train)
    df_train, df_test, d_corr_tr, v_corr_tr = construir_difference_feature(fs_train_rotulado, fs_test)
    
    return {
        "bruto": df_bruto_train,
        "normalizado": df_t1,
        "filtrado": df_t2,
        "fs_train": df_train,
        "fs_test": df_test,
        "cps": cps,
        "corr_data_train": d_corr_tr,
        "corr_val_train": v_corr_tr
    }

def preparar_dados_redes_neurais(seq_len=30):
    """Prepara matrizes 3D para CNN/LSTM sem carregar rótulos de teste."""
    df_train_raw, df_test_raw, _ = importar_dados()
    
    # Pipeline de tratamento
    df_t0, df_ts0 = definir_condicoes_operacao(df_train_raw, df_test_raw)
    df_t1, df_ts1 = normalizar_dados(df_t0, df_ts0, no_ops=1, coluna_condicao='cluster_op')
    df_t2, df_ts2 = filtrar_dados(df_t1, df_ts1)
    
    fs_train, fs_test = extrair_fused_signal(df_t2, df_ts2)
    fs_train_rotulado, _ = rotular_rul_cpd(fs_train)
    df_train, df_test, _, _ = construir_difference_feature(fs_train_rotulado, fs_test)
    
    # Retorna apenas os dados de treino (X, Y) e teste (X)
    return extrair_janelas_3d(df_train, df_test, seq_len)
