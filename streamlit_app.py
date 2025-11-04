# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, validator
from difflib import get_close_matches
import logging
import plotly.express as px

# Configura logging
logging.basicConfig(level=logging.INFO, filename="predizioni.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Classe Input con Pydantic ----------
class InputImmobiliare(BaseModel):
    total_sqft: float = Field(gt=10, lt=10000)
    bath: int = Field(ge=1, le=10)
    balcony: int = Field(ge=0, le=5)
    bhk: int = Field(ge=1, le=10)
    location: str

    @validator("location")
    def clean_location(cls, v):
        return v.strip().lower().replace(" ", "_")

# ---------- Funzione di predizione ----------
def predici_prezzo_immobile(input_data: InputImmobiliare, model, X_columns, default_location='centro', valuta='€'):
    x_input = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    # Popola feature numeriche
    for col in ['total_sqft', 'bath', 'balcony', 'bhk']:
        if col in x_input.columns:
            x_input[col] = getattr(input_data, col)
    # Popola location
    loc_col = 'location_' + input_data.location
    valid_locations = [c for c in X_columns if c.startswith('location_')]
    if loc_col in valid_locations:
        x_input[loc_col] = 1
    else:
        matches = get_close_matches(loc_col, valid_locations, n=1)
        if matches:
            x_input[matches[0]] = 1
            st.warning(f"Location '{input_data.location}' non trovata, usata '{matches[0][9:]}' per similarità.")
        else:
            default_col = 'location_' + default_location
            if default_col in valid_locations:
                x_input[default_col] = 1
                st.warning(f"Location '{input_data.location}' non trovata, usata fallback '{default_location}'.")
            else:
                raise ValueError(f"Location '{input_data.location}' e fallback '{default_location}' non valide.")

    # Predizione con intervallo di confidenza se RandomForest
    if hasattr(model, "estimators_"):
        preds = np.array([tree.predict(x_input)[0] for tree in model.estimators_])
        pred_mean = preds.mean()
        pred_std = preds.std()
        prezzo = round(pred_mean, -2)
        logging.info(f"{input_data} -> {prezzo} ± {round(pred_std,2)}")
        return prezzo, round(pred_std, 2)
    else:
        prezzo = round(model.predict(x_input)[0], -2)
        logging.info(f"{input_data} -> {prezzo}")
        return prezzo, None

# ---------- Dummy Model per esempio ----------
class DummyModel:
    def predict(self, x):
        # Genera predizione randomica intorno a 300k-800k per esempio
        return [np.random.randint(300000, 800000)]

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Predizione Prezzo Immobiliare", layout="wide")
    st.title("🏠 Predizione Prezzo Immobiliare")
    st.markdown("Inserisci le caratteristiche dell'immobile e calcola il prezzo stimato.")

    # Input utente
    col1, col2 = st.columns(2)
    with col1:
        total_sqft = st.slider("Superficie (m²)", 10, 10000, 90)
        bath = st.number_input("Numero bagni", 1, 10, 2)
        balcony = st.number_input("Numero balconi", 0, 5, 1)
    with col2:
        bhk = st.number_input("Numero stanze (BHK)", 1, 10, 3)
        location = st.text_input("Location", "buccinasco")

    # Colonne modello simulate
    X_columns = ['total_sqft', 'bath', 'balcony', 'bhk',
                 'location_centro', 'location_periferia', 'location_buccinasco']

    # Modello dummy
    model = DummyModel()

    if st.button("Calcola prezzo"):
        try:
            input_data = InputImmobiliare(total_sqft=total_sqft, bath=bath, balcony=balcony, bhk=bhk, location=location)
            prezzo, conf = predici_prezzo_immobile(input_data, model, X_columns)
            st.success(f"💰 Prezzo stimato: €{prezzo:,.0f}")
            if conf:
                st.info(f"± intervallo di confidenza stimato: €{conf:,.0f}")
        except ValidationError as ve:
            st.error(f"Errore di validazione: {ve}")
        except Exception as e:
            st.error(f"Errore nella predizione: {e}")

    # Grafico interattivo di esempio con Plotly
    st.subheader("📊 Distribuzione Prezzi Simulata")
    np.random.seed(42)
    sample_prices = np.random.randint(200000, 1000000, size=100)
    df_plot = pd.DataFrame({'Prezzo (€)': sample_prices})
    fig = px.histogram(df_plot, x='Prezzo (€)', nbins=20, title="Distribuzione prezzi immobili (simulata)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
