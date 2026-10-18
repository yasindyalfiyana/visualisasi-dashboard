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

# Nilai Evaluasi Model
mae = 50912.72
rmse = 63448.37
r2 = -1.0002

# --- SIDEBAR: MENU NAVIGASI HALAMAN (BISA DIKLIK) ---
st.sidebar.title("📌 Navigasi Menu")
pilihan_menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Home (Beranda)", "📈 Analisis Tren & Visualisasi", "🎯 Evaluasi & Prediksi Model", "📋 Tabel Data Historis"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Info Skripsi:** Implementasi Random Forest & CRISP-DM.")


# ==========================================
# 1. HALAMAN HOME (BERANDA)
# ==========================================
if pilihan_menu == "🏠 Home (Beranda)":
    st.title("📊 Selamat Datang di Dashboard Analisis & Prediksi Penjualan")
    st.markdown("### Sistem Pendukung Keputusan Berbasis *Machine Learning*")
    
    st.info("""
    Aplikasi web interaktif ini dikembangkan sebagai bagian dari penulisan tugas akhir / skripsi Program Studi Sistem Informasi. 
    Dashboard ini mengimplementasikan kerangka kerja **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 
    untuk menganalisis tren dan memprediksi penjualan kategori **Technology** menggunakan algoritma **Random Forest Regression**.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("📁 Fokus Kategori", "Technology")
    col2.metric("📅 Total Bulan Observasi", f"{len(df_monthly)} Bulan")
    col3.metric("🤖 Algoritma", "Random Forest")

    st.markdown("---")
    st.subheader("🚀 Fitur Utama Dashboard:")
    st.write("1. **Home (Beranda):** Halaman pengantar dan informasi ringkas mengenai penelitian.")
    st.write("2. **Analisis Tren & Visualisasi:** Menampilkan grafik interaktif perkembangan penjualan bulanan.")
    st.write("3. **Evaluasi & Prediksi Model:** Menampilkan metrik evaluasi (MAE, RMSE, $R^2$) serta panel simulasi prediksi.")
    st.write("4. **Tabel Data Historis:** Menyediakan rincian data bersih beserta fitur *lag* yang digunakan.")


# ==========================================
# 2. HALAMAN ANALISIS TREN & VISUALISASI
# ==========================================
elif pilihan_menu == "📈 Analisis Tren & Visualisasi":
    st.title("📈 Analisis Tren Penjualan Kategori Technology")
    st.markdown("Visualisasi grafik interaktif perkembangan total penjualan bulanan berdasarkan data historis.")

    # Ringkasan Metrik Cepat di Atas Grafik
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Penjualan Keseluruhan", f"US$ {df_monthly['Total_Sales'].sum():,.0f}")
    m2.metric("Rata-rata Penjualan / Bulan", f"US$ {df_monthly['Total_Sales'].mean():,.0f}")
    m3.metric("Jumlah Observasi", f"{len(df_monthly)} Bulan")

    st.markdown("---")
    
    # Grafik Batang Interaktif (Tema Biru Profesional)
    fig_bar = px.bar(
        df_monthly, x='Month', y='Total_Sales',
        text_auto='.2s',
        title="Distribusi Total Penjualan Per Bulan (Global Superstore)",
        labels={'Total_Sales': 'Total Penjualan (US$)', 'Month': 'Bulan'}
    )
    fig_bar.update_traces(marker_color='#0066cc', textfont_size=10, textangle=0, textposition="outside")
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# 3. HALAMAN EVALUASI & PREDIKSI MODEL
# ==========================================
elif pilihan_menu == "🎯 Evaluasi & Prediksi Model":
    st.title("🎯 Evaluasi Kinerja Model & Simulasi Prediksi")
    st.markdown("Evaluasi hasil pengujian model *Random Forest* pada data uji (*testing*) serta simulasi prediksi bulanan.")

    # Kartu Metrik Evaluasi
    st.subheader("📉 Metrik Evaluasi Pengujian Model")
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE (Mean Absolute Error)", f"US$ {mae:,.2f}")
    e2.metric("RMSE (Root Mean Squared Error)", f"US$ {rmse:,.2f}")
    e3.metric("R² Score (Determinasi)", f"{r2:.4f}")

    st.markdown("---")
    
    # Grafik Perbandingan Aktual vs Prediksi
    st.subheader("📉 Grafik Perbandingan: Nilai Aktual vs Prediksi Model")
    df_comparison = pd.DataFrame({
        'Aktual': y_test.values,
        'Prediksi Model': y_pred
    }, index=df_model.loc[X_test.index, 'Month']).sort_index()

    df_melted = df_comparison.reset_index().melt(id_vars='Month', value_vars=['Aktual', 'Prediksi Model'], 
                                                 var_name='Tipe', value_name='Nilai Sales')

    fig_comp = px.line(
        df_melted, x='Month', y='Nilai Sales', color='Tipe',
        markers=True,
        title="Perbandingan Nilai Aktual dan Prediksi (Data Uji)",
        color_discrete_map={'Aktual': '#0066cc', 'Prediksi Model': '#00adb5'}
    )
    fig_comp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    
    # Bagian Simulasi Prediksi di Halaman Utama
    st.subheader("⚙️ Panel Simulasi Prediksi Bulanan")
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_lag_1 = st.number_input("Penjualan Bulan Lalu (Lag 1)", value=float(df_monthly['Total_Sales'].iloc[-1]), format="%.2f")
    with col_sim2:
        sim_lag_2 = st.number_input("Penjualan 2 Bulan Lalu (Lag 2)", value=float(df_monthly['Total_Sales'].iloc[-2]), format="%.2f")

    if st.button("🚀 Hitung Prediksi Penjualan", use_container_width=True):
        hasil_sim = model.predict([[sim_lag_1, sim_lag_2]])
        st.success(f"### Hasil Prediksi Penjualan Bulan Berikutnya:\n**US$ {hasil_sim[0]:,.2f}**")


# ==========================================
# 4. HALAMAN TABEL DATA HISTORIS
# ==========================================
elif pilihan_menu == "📋 Tabel Data Historis":
    st.title("📋 Tabel Data Historis & Fitur Rekayasa (*Lag*)")
    st.markdown("Rincian data transaksi bulanan kategori *Technology* beserta atribut *Lag-1* dan *Lag-2* yang digunakan dalam pelatihan model.")
    
    st.dataframe(df_monthly, use_container_width=True)
