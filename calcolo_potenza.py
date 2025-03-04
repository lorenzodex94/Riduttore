import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Titolo dell'app
st.title("Previsione Potenza e Temperatura Forno")

# Percorso del file CSV (modifica con il tuo percorso reale)
FILE_PATH = "Riduttore/Dati riduttore 1.csv"  # Sostituisci con il nome del tuo file CSV

# Caricamento dei dati
@st.cache_data  # Caching per migliorare le prestazioni
def load_data(file_path):
    df = pd.read_csv(file_path)  # Carica il file CSV
    return df


# Prova a caricare i dati. Se fallisce, mostra un messaggio d'errore.
try:
    df = load_data(FILE_PATH)
    st.write("Anteprima dei dati caricati:")
    st.dataframe(df.head())  # Mostra le prime righe del dataframe

    # Definisci le variabili target e le feature
    target_potenza = df[['Media di Somma Potenze elettriche kW']].values.ravel()
    target_temperatura = df[['Media di TEMPERATURA USCITA FORNO INDUCTOTHERM']].values.ravel()
    feature_columns = ['Spessore richiesto [mm]', 'Media di VELOCITA INGRESSO TUBO',
                        'RIFERIMENTO INDUCTOTHERM', 'RIFERIMENTO ZONA 1 ASEA',
                        'RIFERIMENTO ZONA 2 ASEA', 'RIFERIMENTO ZONA 3 ASEA',
                        'Carbonio equivalente']

    X = df[feature_columns]

    # 2. Pulizia dei Nomi delle Colonne (FATTO SOLO UNA VOLTA)
    X.columns = X.columns.str.replace('[', '_', regex=False).str.replace(']', '_', regex=False).str.replace('<', '_', regex=False)
    X.columns = X.columns.str.replace(' ', '_', regex=False)  # Replace spaces as well

    # *** STAMPA DI DEBUG: Verificare i nomi delle colonne ***
    # st.write("Nomi delle colonne dopo la pulizia:", X.columns)


    # 3. Divisione in Train, Validation e Test (FATTA SOLO UNA VOLTA)
    X_train, X_temp, y_train_potenza, y_temp_potenza = train_test_split(X, target_potenza, test_size=0.3, random_state=42)  # random_state fisso
    X_val, X_test, y_val_potenza, y_test_potenza = train_test_split(X_temp, y_temp_potenza, test_size=0.5, random_state=42)  # random_state fisso

    X_train, X_temp, y_train_temperatura, y_temp_temperatura = train_test_split(X, target_temperatura, test_size=0.3, random_state=42)  # random_state fisso
    X_val, X_test, y_val_temperatura, y_test_temperatura = train_test_split(X_temp, y_temp_temperatura, test_size=0.5, random_state=42)  # random_state fisso

    # 4. Primo Modello: Random Forest per la Potenza
    # Scaling delle feature
    scaler_potenza = StandardScaler()
    X_train_scaled = scaler_potenza.fit_transform(X_train)
    X_val_scaled = scaler_potenza.transform(X_val)
    X_test_scaled = scaler_potenza.transform(X_test)

    rf_regressor_potenza = RandomForestRegressor(n_estimators=400, random_state=42)  # random_state fisso
    rf_regressor_potenza.fit(X_train_scaled, y_train_potenza)

    y_pred_potenza = rf_regressor_potenza.predict(X_test_scaled)

    mse_potenza = mean_squared_error(y_test_potenza, y_pred_potenza)
    r2_potenza = r2_score(y_test_potenza, y_pred_potenza)

    st.write(f"Potenza - Mean Squared Error: {mse_potenza}")
    st.write(f"Potenza - R-squared: {r2_potenza}")

    # 5. Crea X2 (Usa le predizioni del primo modello)
    X2 = pd.concat([X, pd.DataFrame(target_potenza, columns=['Stima_potenza_elettrica_reale'])], axis=1)  # Usa valori reali

    # Scala X prima di fare la predizione
    X_scaled = scaler_potenza.transform(X)
    X2['Stima_potenza_elettrica_predetta'] = rf_regressor_potenza.predict(X_scaled)  # Aggiungi la colonna di predizioni

    # 6. Modelli Ensemble per la Temperatura (USANDO X2 come input)
    # Scaling delle feature
    scaler_temperatura = StandardScaler()
    # Dividi X2 (usando *gli stessi* indici dei passaggi precedenti)
    X_train_ensemble = X2.iloc[X_train.index]
    X_val_ensemble = X2.iloc[X_val.index]
    X_test_ensemble = X2.iloc[X_test.index]

    X_train_ensemble_scaled = scaler_temperatura.fit_transform(X_train_ensemble)
    X_val_ensemble_scaled = scaler_temperatura.transform(X_val_ensemble)
    X_test_ensemble_scaled = scaler_temperatura.transform(X_test_ensemble)

    rf_model_temperatura = RandomForestRegressor(n_estimators=400, random_state=42)  # random_state fisso
    xgb_model_temperatura = XGBRegressor(n_estimators=400, random_state=42)  # random_state fisso

    rf_model_temperatura.fit(X_train_ensemble_scaled, y_train_temperatura)
    xgb_model_temperatura.fit(X_train_ensemble_scaled, y_train_temperatura)

    # 7. Predizioni e Combinazione (Val e Test Set)
    rf_predictions_val = rf_model_temperatura.predict(X_val_ensemble_scaled)
    xgb_predictions_val = xgb_model_temperatura.predict(X_val_ensemble_scaled)

    rf_predictions_test = rf_model_temperatura.predict(X_test_ensemble_scaled)
    xgb_predictions_test = xgb_model_temperatura.predict(X_test_ensemble_scaled)

    # 8. Stacking (Meta-Modello) - CORRETTO: Addestramento su Validation Set

    stacked_predictions_val = np.column_stack((rf_predictions_val, xgb_predictions_val))
    stacked_predictions_test = np.column_stack((rf_predictions_test, xgb_predictions_test))

    meta_model_temperatura = LinearRegression()
    meta_model_temperatura.fit(stacked_predictions_val, y_val_temperatura)  # Addestramento su *VALIDATION* set

    final_predictions_temperatura = meta_model_temperatura.predict(stacked_predictions_test)  # Predizione su test set

    # 9. Valutazione Finale (Solo su Test Set)

    mse_temperatura = mean_squared_error(y_test_temperatura, final_predictions_temperatura)
    r2_temperatura = r2_score(y_test_temperatura, final_predictions_temperatura)
    mape_temperatura = mean_absolute_percentage_error(y_test_temperatura, final_predictions_temperatura) * 100
    mae_temperatura = mean_absolute_error(y_test_temperatura, final_predictions_temperatura)

    st.write(f'Temperatura - Mean Squared Error: {mse_temperatura}')
    st.write(f'Temperatura - R-squared: {r2_temperatura}')
    st.write(f'Temperatura - Mean Absolute Percentage Error (MAPE): {mape_temperatura}%')
    st.write(f'Temperatura - Mean Absolute Error: {mae_temperatura}')

    # 10. Visualizzazione (Opzionale)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_temperatura, final_predictions_temperatura, alpha=0.7)
    ax.plot([min(y_test_temperatura), max(y_test_temperatura)], [min(y_test_temperatura), max(y_test_temperatura)],
            linestyle='--', color='red', label='Linea di Perfetta Predizione')
    ax.set_xlabel('Valori Reali (Temperatura)')
    ax.set_ylabel('Predizioni (Temperatura)')
    ax.set_title('Valori Reali vs. Predizioni (Meta-Modello)')
    ax.legend()
    st.pyplot(fig)

    # 11. Interfaccia Interattiva

    st.header("Inserisci i parametri per la predizione:")

    # Creazione degli slider in Streamlit
    spessore = st.slider('Spessore [mm]:', min_value=float(df['Spessore richiesto [mm]'].min()),
                         max_value=float(df['Spessore richiesto [mm]'].max()),
                         value=float(df['Spessore richiesto [mm]'].mean()), step=0.1)
    velocita = st.slider('Velocità:', min_value=float(df['Media di VELOCITA INGRESSO TUBO'].min()),
                         max_value=float(df['Media di VELOCITA INGRESSO TUBO'].max()),
                         value=float(df['Media di VELOCITA INGRESSO TUBO'].mean()), step=1.0)
    inductotherm = st.slider('Inductotherm:', min_value=float(df['RIFERIMENTO INDUCTOTHERM'].min()),
                             max_value=float(df['RIFERIMENTO INDUCTOTHERM'].max()),
                             value=float(df['RIFERIMENTO INDUCTOTHERM'].mean()), step=1.0)
    zona1 = st.slider('Zona 1:', min_value=float(df['RIFERIMENTO ZONA 1 ASEA'].min()),
                       max_value=float(df['RIFERIMENTO ZONA 1 ASEA'].max()),
                       value=float(df['RIFERIMENTO ZONA 1 ASEA'].mean()), step=1.0)
    zona2 = st.slider('Zona 2:', min_value=float(df['RIFERIMENTO ZONA 2 ASEA'].min()),
                       max_value=float(df['RIFERIMENTO ZONA 2 ASEA'].max()),
                       value=float(df['RIFERIMENTO ZONA 2 ASEA'].mean()), step=1.0)
    zona3 = st.slider('Zona 3:', min_value=float(df['RIFERIMENTO ZONA 3 ASEA'].min()),
                       max_value=float(df['RIFERIMENTO ZONA 3 ASEA'].max()),
                       value=float(df['RIFERIMENTO ZONA 3 ASEA'].mean()), step=1.0)
    carbonio = st.slider('Carbonio:', min_value=float(df['Carbonio equivalente'].min()),
                         max_value=float(df['Carbonio equivalente'].max()),
                         value=float(df['Carbonio equivalente'].mean()), step=0.01)


    # Funzione di predizione
    def predict_values(spessore, velocita, inductotherm, zona1, zona2, zona3, carbonio):
        """Funzione per fare predizioni con i valori degli slider."""

        # *** STAMPA DI DEBUG: Valori ricevuti dalla funzione ***
        st.write("Valori ricevuti dalla funzione predict_values:")
        st.write(f"  Spessore: {spessore:.2f} [mm]")
        st.write(f"  Velocità: {velocita:.2f} [m/min]")
        st.write(f"  Inductotherm: {inductotherm:.2f}")
        st.write(f"  Zona1: {zona1:.2f}")
        st.write(f"  Zona2: {zona2:.2f}")
        st.write(f"  Zona3: {zona3:.2f}")
        st.write(f"  Carbonio: {carbonio:.2f}")
        st.write(f"  Produttività: {(3.14 / 4 * (0.101 ** 2 - (0.101 - 2 * (spessore / 1000)) ** 2) * (
                    velocita / 60) * 7850 * 3.6):.2f} [ton/h]")

        # Crea un DataFrame con i valori inseriti
        # *** USA IL NOME ESATTO DELLA COLONNA TROVATO CON X.columns ***
        input_data = pd.DataFrame({
            X.columns[0]: [spessore],  # Assumendo che 'Spessore' sia la prima colonna
            X.columns[1]: [velocita],
            X.columns[2]: [inductotherm],
            X.columns[3]: [zona1],
            X.columns[4]: [zona2],
            X.columns[5]: [zona3],
            X.columns[6]: [carbonio]
        })

        # Scaling dei dati di input
        input_data_scaled = scaler_potenza.transform(input_data)

        # Predizione della potenza
        potenza_predetta = rf_regressor_potenza.predict(input_data_scaled)[0]

        # Crea X2 per la predizione della temperatura (come nel training)
        input_data_X2 = pd.concat([input_data, pd.DataFrame({'Stima_potenza_elettrica_reale': [0]}), ], axis=1)  # Valore reale non usato qui

        # Prevedi la potenza e aggiungila a X2
        input_data_scaled_for_potenza = scaler_potenza.transform(input_data)
        input_data_X2['Stima_potenza_elettrica_predetta'] = rf_regressor_potenza.predict(input_data_scaled_for_potenza)

        # Scaling e predizione della temperatura
        input_data_X2_scaled = scaler_temperatura.transform(input_data_X2)
        rf_prediction = rf_model_temperatura.predict(input_data_X2_scaled)
        xgb_prediction = xgb_model_temperatura.predict(input_data_X2_scaled)
        stacked_input = np.column_stack((rf_prediction, xgb_prediction))
        temperatura_predetta = meta_model_temperatura.predict(stacked_input)[0]

        st.write(f"Potenza Elettrica Predetta: {potenza_predetta:.2f} kW")
        st.write(f"Temperatura Predetta: {temperatura_predetta:.2f} °C")


    # Bottone per lanciare la predizione
    if st.button("Predici"):
        predict_values(spessore, velocita, inductotherm, zona1, zona2, zona3, carbonio)

except FileNotFoundError:
    st.error(f"Errore: Impossibile trovare il file CSV. Assicurati che il file '{FILE_PATH}' si trovi nella stessa directory dello script e che il nome del file sia corretto.")
except Exception as e:
    st.error(f"Si è verificato un errore durante il caricamento dei dati: {e}")
