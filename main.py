import pandas as pd
import pmdarima as pm
from pmdarima.model_selection import train_test_split

def read_csv(file_path):

    df = pd.read_csv(file_path, delimiter=",", header=0)

    df["mes_ref"] = pd.to_datetime(df["mes_ref"])

    df.set_index("mes_ref", inplace=True)

    return df


# DataFrame variáveis
var_cambio = read_csv("data/variaveisCambio.csv")

# DataFrame volume de câmbio
volume = read_csv("data/volumeCambio.csv")

df = volume.merge(var_cambio, on="mes_ref")

# carrega as colunas necessárias dos arquivos de projeção
df_proj = read_csv("data/projecaoBase.csv")
df_projg = read_csv("data/projecaoOtimi.csv")
df_projb = read_csv("data/projecaoPessim.csv")


# Separa em treino e teste
train, test = train_test_split(df, test_size=11)

# Series para Np-Array e dropando colunas de endógenas
exog = train.drop(columns=["volume_exp", "volume_imp"]).values.reshape(-1, 6)

# Estruturando o modelo SARIMAX
model = pm.auto_arima(
    y=train["volume_exp"],
    X=exog,
    d=1,
    D=1,
    max_order=21,
    m=12,
    seasonal=True,
    error_action="warn",
    suppress_warnings=True,
    stepwise=True,
    n_fits=50,
    random_state=42,
    method='lbfgs',
    information_criterion='bic'
)
# Numero de meses a ser previsto
n_periods = 11

# Separando os dados treinado e intervalo de confidence
forecast  = model.predict(n_periods=n_periods, X=df_proj)

# Definindo o index
index_fc = pd.date_range(df.index[-1], periods=n_periods, freq="MS")

# Transformando os dados para Series
forecast_series = pd.Series(forecast , index=index_fc)  # Dados preditados

# Summary do modelo
print(forecast_series)
