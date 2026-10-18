import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Mempersiapkan fitur Lag dan Melakukan Split Data (80% Training, 20% Testing) sesuai eksperimen
if 'Lag_1' not in df_monthly.columns or 'Lag_2' not in df_monthly.columns:
    df_monthly['Lag_1'] = df_monthly['Total_Sales'].shift(1)
    df_monthly['Lag_2'] = df_monthly['Total_Sales'].shift(2)

df_model = df_monthly.dropna().copy()
X = df_model[['Lag_1', 'Lag_2']]
y = df_model['Total_Sales']

# Menggunakan random_state yang konsisten (misal: 42) untuk pembagian data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# Menghitung Metrik Evaluasi pada Data Testing
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 1. Metrik Ringkasan Utama
col1, col2, col3 = st.columns(3)
col1.metric("Total Bulan Observasi", f"{len(df_monthly)} Bulan")
col2.metric("Total Penjualan Keseluruhan", f"US$ {df_monthly['Total_Sales'].sum():,.2f}")
col3.metric("Rata-rata Penjualan Bulanan", f"US$ {df_monthly['Total_Sales'].mean():,.2f}")

st.markdown("---")

# 2. Ringkasan Hasil Evaluasi Model (Sesuai Pengujian Bab IV)
st.subheader("🎯 Hasil Evaluasi Kinerja Model Random Forest (Data Testing)")
ev_col1, ev_col2, ev_col3 = st.columns(3)
ev_col1.metric("MAE (Mean Absolute Error)", f"US$ {mae:,.2f}")
ev_col2.metric("RMSE (Root Mean Squared Error)", f"US$ {rmse:,.2f}")
ev_col3.metric("R² Score (Koefisien Determinasi)", f"{r2:.4f}")

st.markdown("---")

# 3. Visualisasi Grafik Tren Penjualan Bulanan
st.subheader("📈 Grafik Tren Penjualan Bulanan (Global Superstore)")
st.line_chart(df_monthly.set_index('Month')['Total_Sales'])

# 4. Grafik Perbandingan Aktual vs Prediksi pada Data Testing
st.subheader("📉 Perbandingan Nilai Penjualan Aktual vs Hasil Prediksi (Data Testing)")
df_comparison = pd.DataFrame({
    'Aktual': y_test.values,
    'Prediksi Model': y_pred
}, index=df_model.loc[X_test.index, 'Month'])

# Urutkan berdasarkan tanggal agar grafik garis tersambung rapi
df_comparison = df_comparison.sort_index()
st.line_chart(df_comparison)

# 5. Menampilkan Tabel Data Historis & Fitur Lag
st.subheader("📋 Tabel Data Historis & Rekayasa Fitur (Lag-1, Lag-2)")
st.dataframe(df_monthly, use_container_width=True)

# 6. Fitur Simulasi Prediksi Nilai Penjualan
st.sidebar.header("⚙️ Panel Simulasi Prediksi")
st.sidebar.write("Masukkan nilai penjualan historis sebelumnya untuk memprediksi bulan berikutnya:")

input_lag_1 = st.sidebar.number_input("Penjualan Bulan Lalu (Lag 1)", value=float(df_monthly['Total_Sales'].iloc[-1]))
input_lag_2 = st.sidebar.number_input("Penjualan 2 Bulan Lalu (Lag 2)", value=float(df_monthly['Total_Sales'].iloc[-2]))

if st.sidebar.button("Prediksi Penjualan"):
    pred_result = model.predict([[input_lag_1, input_lag_2]])
    st.sidebar.success(f"Hasil Prediksi Penjualan: **US$ {pred_result[0]:,.2f}**")
