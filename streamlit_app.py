import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, validator
from difflib import get_close_matches
import logging
import plotly.express as px

# Configurazione logging
logging.basicConfig(level=logging.INFO, filename="predizioni.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

class InputImmobiliare(BaseModel):
    total_sqft: float = Field(gt=10, lt=10000)
    bath: int = Field(ge=1, le=10)
    balcony: int = Field(ge=0, le=5)
    bhk: int = Field(ge=1, le=10)
    location: str

    @validator("location")
    def clean_location(cls, v):
        return v.strip().lower().replace(" ", "_")

def predici_prezzo_immobile(input_data: InputImmobiliare, model, X_columns, default_location='centro'):
    x_input = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    
    for col in ['total_sqft', 'bath', 'balcony', 'bhk']:
        if col in x_input.columns:
            x_input[col] = getattr(input_data, col)
    
    loc_col = 'location_' + input_data.location
    valid_locations = [c for c in X_columns if c.startswith('location_')]
    
    # Return codice location usato, messaggio warning se serve
    code_location_usata = None
    warning_message = None

    if loc_col in valid_locations:
        x_input[loc_col] = 1
        code_location_usata = input_data.location
    else:
        matches = get_close_matches(loc_col, valid_locations, n=1)
        if matches:
            x_input[matches[0]] = 1
            code_location_usata = matches[0][9:]  # toglie 'location_'
            warning_message = f"Location '{input_data.location}' non trovata, usata '{code_location_usata}' per similarità."
        else:
            default_col = 'location_' + default_location
            if default_col in valid_locations:
                x_input[default_col] = 1
                code_location_usata = default_location
                warning_message = f"Location '{input_data.location}' non trovata, usata fallback '{default_location}'."
            else:
                raise ValueError(f"Location '{input_data.location}' e fallback '{default_location}' non valide.")
    
    # Predizione con intervallo di confidenza se modello ensemble
    if hasattr(model, "estimators_"):
        preds = np.array([tree.predict(x_input)[0] for tree in model.estimators_])
        pred_mean = preds.mean()
        pred_std = preds.std()
        prezzo = round(pred_mean, -2)
        logging.info(f"{input_data} -> {prezzo} ± {round(pred_std, 2)}")
        return prezzo, round(pred_std, 2), warning_message, code_location_usata
    else:
        prezzo = round(model.predict(x_input)[0], -2)
        logging.info(f"{input_data} -> {prezzo}")
        return prezzo, None, warning_message, code_location_usata

def main():
    st.set_page_config(page_title="Predizione Prezzo Immobiliare", layout="wide")
    st.title("🏠 Predizione Prezzo Immobiliare")
    st.markdown("Inserisci le caratteristiche dell'immobile e calcola il prezzo stimato.")

    col1, col2 = st.columns(2)
    with col1:
        total_sqft = st.slider("Superficie (m²)", 10, 10000, 90)
        bath = st.number_input("Numero bagni", 1, 10, 2)
        balcony = st.number_input("Numero balconi", 0, 5, 1)
    with col2:
        bhk = st.number_input("Numero stanze (BHK)", 1, 10, 3)
        location = st.text_input("Location", "buccinasco")

    X_columns = ['total_sqft', 'bath', 'balcony', 'bhk',
                 'location_centro', 'location_periferia', 'location_buccinasco']

    class DummyModel:
        def predict(self, x):
            return [np.random.randint(300000, 800000)]

    model = DummyModel()

    if st.button("Calcola prezzo"):
        try:
            input_data = InputImmobiliare(total_sqft=total_sqft, bath=bath, balcony=balcony, bhk=bhk, location=location)
            prezzo, conf, warning, loc_usata = predici_prezzo_immobile(input_data, model, X_columns)
            if warning:
                st.warning(warning)
            st.success(f"💰 Prezzo stimato: €{prezzo:,.0f}")
            if conf:
                st.info(f"± intervallo di confidenza stimato: €{conf:,.0f}")
        except ValidationError as ve:
            st.error(f"Errore di validazione: {ve}")
        except Exception as e:
            st.error(f"Errore nella predizione: {e}")

    st.subheader("📊 Distribuzione Prezzi Simulata")
    np.random.seed(42)
    sample_prices = np.random.randint(200000, 1000000, size=100)
    df_plot = pd.DataFrame({'Prezzo (€)': sample_prices})
    fig = px.histogram(df_plot, x='Prezzo (€)', nbins=20, title="Distribuzione prezzi immobili (simulata)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
