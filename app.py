
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report

# Configuración de página estilo minimalista
st.set_page_config(page_title="Credit Risk Analytics", layout="wide")

# Estilos CSS personalizados para reforzar el diseño editorial de Floema
st.markdown("""
    <style>
    .reportview-container .main .block-container{ max-width: 1200px; }
    h1, h2, h3 { color: #364A3F !important; font-family: 'Playfair Display', serif; }
    div.stButton > button:first-child {
        background-color: #364A3F; color: white; border-radius: 4px; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 Credit Risk & Market Scoring App")
st.caption("Diseño inspirado en Floema — Flujo de Machine Learning para Riesgo Crediticio")
st.write("---")

# ----------------- SIDEBAR: NAVEGACIÓN -----------------
st.sidebar.header("Navegación del Flujo")
menu = st.sidebar.radio(
    "Selecciona un paso:",
    ["1. Carga de Datos", "2. Limpieza de Datos", "3. Exploración EDA", "4. Regresión Logística", "5. Random Forest", "6. Descarga de Modelo"]
)

# Inicializar estados de sesión para no perder la información al cambiar de pestaña
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'model_lr' not in st.session_state:
    st.session_state.model_lr = None
if 'model_rf' not in st.session_state:
    st.session_state.model_rf = None

# ----------------- PASO 1: CARGA DE DATOS -----------------
if menu == "1. Carga de Datos":
    st.header("📥 Carga de Dataset")
    st.write("Sube el archivo CSV de riesgo crediticio para iniciar el flujo.")
    
    uploaded_file = st.file_uploader("Elige tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("¡Archivo cargado con éxito!")
    
    if st.session_state.df is not None:
        st.subheader("Vista previa de los datos originales")
        st.dataframe(st.session_state.df.head(10))

# ----------------- PASO 2: LIMPIEZA DE DATOS -----------------
elif menu == "2. Limpieza de Datos":
    st.header("🧼 Limpieza de Datos Atípicos")
    
    if st.session_state.df is None:
        st.warning("⚠️ Por favor, primero carga un archivo en el Paso 1.")
    else:
        st.write("En este paso removemos el error de registros de antigüedad laboral mayores a 60 años y separamos el target.")
        
        # Proceso de limpieza idéntico a tu Colab
        df_clean = st.session_state.df.copy()
        if 'person_emp_length' in df_clean.columns:
            df_clean = df_clean[df_clean['person_emp_length'] < 60]
        
        st.session_state.df_clean = df_clean
        st.success("¡Limpieza ejecutada!")
        
        st.subheader("Dataset Filtrado (Sin Outliers)")
        st.dataframe(st.session_state.df_clean.head(10))

# ----------------- PASO 3: EXPLORACIÓN EDA -----------------
elif menu == "3. Exploración EDA":
    st.header("📊 Exploración de Datos (EDA)")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ Por favor, pasa por el paso de Limpieza primero.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución de Estados del Préstamo (Target)")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(data=st.session_state.df_clean, x='loan_status', palette=['#364A3F', '#EAE6DF'], ax=ax)
            ax.set_xticklabels(['No Default (0)', 'Default (1)'])
            st.pyplot(fig)
            
        with col2:
            st.subheader("Antigüedad Laboral vs Monto del Préstamo")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.scatterplot(data=st.session_state.df_clean, x='person_emp_length', y='loan_amnt', hue='loan_status', palette=['#364A3F', '#C2593F'], alpha=0.6, ax=ax)
            st.pyplot(fig)

# ----------------- PASO 4: REGRESIÓN LOGÍSTICA -----------------
elif menu == "4. Regresión Logística":
    st.header("📈 Modelo 1: Regresión Logística")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ Requiere datos limpios del Paso 2.")
    else:
        # Preparación de variables de tu Colab
        X = st.session_state.df_clean.drop(columns=['loan_status', 'person_age', 'person_home_ownership', 'loan_intent', 'loan_grade'], errors='ignore')
        y = st.session_state.df_clean['loan_status']
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        if st.button("Entrenar Regresión Logística"):
            model_lr = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
            model_lr.fit(X_train, y_train)
            st.session_state.model_lr = model_lr
            st.success("¡Modelo entrenado con éxito!")
            
        if st.session_state.model_lr is not None:
            y_pred = st.session_state.model_lr.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            st.subheader("Métricas de Rendimiento")
            st.text(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
            
            st.subheader("Matriz de Confusión")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'], ax=ax)
            st.pyplot(fig)

# ----------------- PASO 5: RANDOM FOREST -----------------
elif menu == "5. Random Forest":
    st.header("🌲 Modelo 2: Random Forest Classifier")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ Requiere datos limpios del Paso 2.")
    else:
        X = st.session_state.df_clean.drop(columns=['loan_status', 'person_age', 'person_home_ownership', 'loan_intent', 'loan_grade'], errors='ignore')
        y = st.session_state.df_clean['loan_status']
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        if st.button("Entrenar Random Forest"):
            model_rf = make_pipeline(SimpleImputer(strategy='mean'), RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            model_rf.fit(X_train, y_train)
            st.session_state.model_rf = model_rf
            st.success("¡Random Forest entrenado con éxito!")
            
        if st.session_state.model_rf is not None:
            y_pred_rf = st.session_state.model_rf.predict(X_test)
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            
            st.subheader("Métricas de Rendimiento (Árboles)")
            st.text(classification_report(y_test, y_pred_rf, target_names=['No Default', 'Default']))
            
            st.subheader("Matriz de Confusión (Random Forest)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'], ax=ax)
            st.pyplot(fig)

# ----------------- PASO 6: DESCARGA DE MODELO -----------------
elif menu == "6. Descarga de Modelo":
    st.header("💾 Exportación y Descarga (.pkl)")
    st.write("Descarga los modelos empaquetados directamente a tu ordenador.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.model_lr is not None:
            joblib.dump(st.session_state.model_lr, 'modelo_regresion_logistica.pkl')
            with open("modelo_regresion_logistica.pkl", "rb") as f:
                st.download_button("Descargar Regresión Logística", f, file_name="modelo_regresion_logistica.pkl")
        else:
            st.info("Entrena la Regresión Logística primero para habilitar la descarga.")
            
    with col2:
        if st.session_state.model_rf is not None:
            joblib.dump(st.session_state.model_rf, 'modelo_random_forest.pkl')
            with open("modelo_random_forest.pkl", "rb") as f:
                st.download_button("Descargar Random Forest", f, file_name="modelo_random_forest.pkl")
        else:
            st.info("Entrena el Random Forest primero para habilitar la descarga.")
