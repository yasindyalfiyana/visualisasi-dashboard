import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman Dashboard
st.set_page_config(
    page_title="Prediksi Penjualan Technology - Random Forest",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Tren & Prediksi Penjualan Kategori Technology")
st.write("Implementasi Algoritma **Random Forest Regression** Berbasis Framework CRISP-DM.")

# Memuat Data Bersih dan Model Machine Learning yang sudah dilatih
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('df_monthly_clean.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    model = joblib.load('model_random_forest.pkl')
    return df, model

df_monthly, model = load_data_and_model()

# 1. Metrik Ringkasan Utama
col1, col2, col3 = st.columns(3)
col1.metric("Total Bulan Observasi", f"{len(df_monthly)} Bulan")
col2.metric("Total Penjualan Keseluruhan", f"US$ {df_monthly['Total_Sales'].sum():,.2f}")
col3.metric("Rata-rata Penjualan Bulanan", f"US$ {df_monthly['Total_Sales'].mean():,.2f}")

st.markdown("---")

# 2. Visualisasi Grafik Tren Penjualan Bulanan
st.subheader("📈 Grafik Tren Penjualan Bulanan (Global Superstore)")
st.line_chart(df_monthly.set_index('Month')['Total_Sales'])

# 3. Menampilkan Tabel Data Historis & Fitur Lag
st.subheader("📋 Tabel Data Historis & Rekayasa Fitur (Lag-1, Lag-2)")
st.dataframe(df_monthly, use_container_width=True)

# 4. Fitur Simulasi Prediksi Nilai Penjualan
st.sidebar.header("⚙️ Panel Simulasi Prediksi")
st.sidebar.write("Masukkan nilai penjualan historis sebelumnya untuk memprediksi bulan berikutnya:")

input_lag_1 = st.sidebar.number_input("Penjualan Bulan Lalu (Lag 1)", value=float(df_monthly['Total_Sales'].iloc[-1]))
input_lag_2 = st.sidebar.number_input("Penjualan 2 Bulan Lalu (Lag 2)", value=float(df_monthly['Total_Sales'].iloc[-2]))

if st.sidebar.button("Prediksi Penjualan"):
    pred_result = model.predict([[input_lag_1, input_lag_2]])
    st.sidebar.success(f"Hasil Prediksi Penjualan: **US$ {pred_result[0]:,.2f}**")
