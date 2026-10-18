import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split

# Konfigurasi Halaman Dashboard (Layout Lebar)
st.set_page_config(
    page_title="Dashboard Analisis & Prediksi Penjualan Technology",
    page_icon="📊",
    layout="wide"
)

# Custom Styling CSS untuk Kartu Metrik Profesional (Tema Tech & Data Science)
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Utama Dashboard
st.title("📊 Dashboard Analisis Tren & Prediksi Penjualan Kategori Technology")
st.markdown("Implementasi Algoritma **Random Forest Regression** Berbasis Framework **CRISP-DM**")

# Memuat Data Bersih dan Model Machine Learning
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('df_monthly_clean.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    model = joblib.load('model_random_forest.pkl')
    return df, model

df_monthly, model = load_data_and_model()

# Persiapan Data & Split untuk Evaluasi
if 'Lag_1' not in df_monthly.columns or 'Lag_2' not in df_monthly.columns:
    df_monthly['Lag_1'] = df_monthly['Total_Sales'].shift(1)
    df_monthly['Lag_2'] = df_monthly['Total_Sales'].shift(2)

df_model = df_monthly.dropna().copy()
X = df_model[['Lag_1', 'Lag_2']]
y = df_model['Total_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# Nilai Evaluasi Model (Sesuai Skripsi)
mae = 50912.72
rmse = 63448.37
r2 = -1.0002

# --- BARIS 1: KARTU METRIK UTAMA ---
st.markdown("### 📌 Ringkasan Kinerja Penjualan")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="📁 Kategori Fokus", value="Technology")
with col2:
    st.metric(label="📅 Total Bulan Observasi", value=f"{len(df_monthly)} Bulan")
with col3:
    st.metric(label="💰 Total Penjualan Keseluruhan", value=f"US$ {df_monthly['Total_Sales'].sum():,.0f}")
with col4:
    st.metric(label="📈 Rata-rata Penjualan / Bulan", value=f"US$ {df_monthly['Total_Sales'].mean():,.0f}")

st.markdown("---")

# --- BARIS 2: GRAFIK TREN BULANAN & EVALUASI MODEL ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Tren Penjualan Bulanan (Bar Chart)")
    # Menggunakan warna Biru Profesional (Tech/Data Science)
    fig_bar = px.bar(
        df_monthly, x='Month', y='Total_Sales',
        text_auto='.2s',
        title="Distribusi Total Penjualan Per Bulan",
        labels={'Total_Sales': 'Total Penjualan (US$)', 'Month': 'Bulan'}
    )
    fig_bar.update_traces(marker_color='#0066cc', textfont_size=10, textangle=0, textposition="outside")
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("🎯 Evaluasi Kinerja Model (Testing)")
    # Kotak metrik evaluasi model
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE", f"US$ {mae:,.0f}")
    e2.metric("RMSE", f"US$ {rmse:,.0f}")
    e3.metric("R² Score", f"{r2:.4f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Grafik Perbandingan Aktual vs Prediksi
    df_comparison = pd.DataFrame({
        'Aktual': y_test.values,
        'Prediksi Model': y_pred
    }, index=df_model.loc[X_test.index, 'Month']).sort_index()

    df_melted = df_comparison.reset_index().melt(id_vars='Month', value_vars=['Aktual', 'Prediksi Model'], 
                                                 var_name='Tipe', value_name='Nilai Sales')

    fig_comp = px.line(
        df_melted, x='Month', y='Nilai Sales', color='Tipe',
        markers=True,
        title="Aktual vs Prediksi (Data Uji)",
        color_discrete_map={'Aktual': '#0066cc', 'Prediksi Model': '#00adb5'}
    )
    fig_comp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# --- BARIS 3: TABEL DATA & PANEL SIMULASI PREDIKSI DI SIDEBAR ---
st.subheader("📋 Tabel Data Historis & Fitur Lag")
st.dataframe(df_monthly, use_container_width=True)

# Sidebar untuk Simulasi Interaktif
st.sidebar.header("⚙️ Panel Simulasi Prediksi")
st.sidebar.markdown("Masukkan nilai historis bulanan untuk memprediksi bulan berikutnya:")

input_lag_1 = st.sidebar.number_input("Penjualan Bulan Lalu (Lag 1)", value=float(df_monthly['Total_Sales'].iloc[-1]), format="%.2f")
input_lag_2 = st.sidebar.number_input("Penjualan 2 Bulan Lalu (Lag 2)", value=float(df_monthly['Total_Sales'].iloc[-2]), format="%.2f")

if st.sidebar.button("🚀 Jalankan Prediksi Model", use_container_width=True):
    pred_result = model.predict([[input_lag_1, input_lag_2]])
    st.sidebar.success(f"### Hasil Prediksi:\n**US$ {pred_result[0]:,.2f}**")

st.sidebar.markdown("---")
st.sidebar.info("💡 Dashboard ini dirancang khusus untuk memenuhi standar visualisasi laporan skripsi berbasis CRISP-DM.")
