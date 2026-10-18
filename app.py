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

# Custom CSS untuk membuat tampilan seperti Aplikasi Dashboard Profesional (Card Style)
st.markdown("""
    <style>
    /* Mengubah warna latar belakang utama */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Styling untuk Menu Sidebar */
    .stRadio label p {
        font-size: 17px !important;
        font-weight: 600 !important;
    }
    
    /* Styling Kotak Kartu (Card Container) */
    .card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    
    /* Perbesar teks judul kartu */
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

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

# --- SIDEBAR: NAVIGASI MENU ---
st.sidebar.title("📌 Navigasi Menu")
pilihan_menu = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Home (Beranda)", 
    "📈 Analisis Tren & Visualisasi", 
    "🎯 Evaluasi & Prediksi Model", 
    "📋 Tabel Data Historis"
])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Sistem Pendukung Keputusan:** Random Forest & CRISP-DM.")


# ==========================================
# 1. HALAMAN HOME (BERANDA)
# ==========================================
if pilihan_menu == "🏠 Home (Beranda)":
    st.title("📊 Dashboard Analisis Tren & Prediksi Penjualan")
    st.markdown("##### *Sistem Pendukung Keputusan Berbasis Machine Learning*")
    
    st.markdown("""
    <div class="card">
        <h4>👋 Selamat Datang di Sistem Visualisasi Skripsi</h4>
        <p>Aplikasi web interaktif ini dikembangkan untuk menganalisis tren penjualan historis dan melakukan peramalan (forecasting) pada kategori <b>Technology</b> menggunakan algoritma <b>Random Forest Regression</b> berdasarkan tahapan framework <b>CRISP-DM</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    # Kotak Metrik Ringkas dalam Card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📁 Fokus Kategori", value="Technology")
    with col2:
        st.metric(label="📅 Total Observasi", value=f"{len(df_monthly)} Bulan")
    with col3:
        st.metric(label="🤖 Algoritma", value="Random Forest")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Identitas Peneliti dalam Card
    st.markdown("""
    <div class="card">
        <h4 style="margin-bottom: 15px;">🎓 Identitas Penelitian & Peneliti</h4>
        <hr style="margin: 5px 0 15px 0;">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <p><b>Nama Mahasiswa:</b> Yasindy Alfiyana</p>
                <p><b>NIM:</b> 43230168</p>
                <p><b>Program Studi:</b> Strata 1 (S1) Sistem Informasi</p>
            </div>
            <div>
                <p><b>Perguruan Tinggi:</b> STMIK IKMI Cirebon</p>
                <p><b>Pembimbing I:</b> Khaerul Anam, M.Kom.</p>
                <p><b>Pembimbing II:</b> Umi Hayati, M.Kom.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 2. HALAMAN ANALISIS TREN & VISUALISASI
# ==========================================
elif pilihan_menu == "📈 Analisis Tren & Visualisasi":
    st.title("📈 Analisis Tren Penjualan Bulanan")
    st.markdown("Visualisasi grafik interaktif perkembangan transaksi kategori *Technology*.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Penjualan Keseluruhan", f"US$ {df_monthly['Total_Sales'].sum():,.0f}")
    m2.metric("Rata-rata Penjualan / Bulan", f"US$ {df_monthly['Total_Sales'].mean():,.0f}")
    m3.metric("Jumlah Bulan Observasi", f"{len(df_monthly)} Bulan")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bungkus Grafik di dalam Card Putih
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig_bar = px.bar(
        df_monthly, x='Month', y='Total_Sales',
        text_auto='.2s',
        title="Distribusi Total Penjualan Per Bulan (Global Superstore)",
        labels={'Total_Sales': 'Total Penjualan (US$)', 'Month': 'Bulan'}
    )
    fig_bar.update_traces(marker_color='#0066cc', textfont_size=10, textangle=0, textposition="outside")
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.info("💡 **Analisis Tren:** Grafik di atas menunjukkan pola fluktuasi historis bulanan yang menjadi dasar pembelajaran bagi model *Machine Learning*.")


# ==========================================
# 3. HALAMAN EVALUASI & PREDIKSI MODEL
# ==========================================
elif pilihan_menu == "🎯 Evaluasi & Prediksi Model":
    st.title("🎯 Evaluasi Model & Simulasi Prediksi")
    st.markdown("Pengujian performa model *Random Forest* dan panel simulasi interaktif.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📉 Metrik Evaluasi Pengujian Model (Data Testing)")
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE (Mean Absolute Error)", f"US$ {mae:,.2f}")
    e2.metric("RMSE (Root Mean Squared Error)", f"US$ {rmse:,.2f}")
    e3.metric("R² Score (Determinasi)", f"{r2:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Bungkus Grafik Komparasi dalam Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📉 Grafik Perbandingan: Aktual vs Prediksi Model")
    df_comparison = pd.DataFrame({
        'Aktual': y_test.values,
        'Prediksi Model': y_pred
    }, index=df_model.loc[X_test.index, 'Month']).sort_index()

    df_melted = df_comparison.reset_index().melt(id_vars='Month', value_vars=['Aktual', 'Prediksi Model'], 
                                                 var_name='Tipe', value_name='Nilai Sales')

    fig_comp = px.line(
        df_melted, x='Month', y='Nilai Sales', color='Tipe',
        markers=True,
        title="Perbandingan Nilai Aktual dan Prediksi pada Data Uji",
        color_discrete_map={'Aktual': '#0066cc', 'Prediksi Model': '#00adb5'}
    )
    fig_comp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Panel Simulasi di dalam Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Panel Simulasi Prediksi Penjualan Bulanan")
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_lag_1 = st.number_input("Penjualan Bulan Lalu (Lag 1)", value=float(df_monthly['Total_Sales'].iloc[-1]), format="%.2f")
    with col_sim2:
        sim_lag_2 = st.number_input("Penjualan 2 Bulan Lalu (Lag 2)", value=float(df_monthly['Total_Sales'].iloc[-2]), format="%.2f")

    if st.button("🚀 Hitung Prediksi Penjualan", use_container_width=True):
        hasil_sim = model.predict([[sim_lag_1, sim_lag_2]])
        st.success(f"### Hasil Prediksi Penjualan Bulan Berikutnya:\n**US$ {hasil_sim[0]:,.2f}**")
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 4. HALAMAN TABEL DATA HISTORIS
# ==========================================
elif pilihan_menu == "📋 Tabel Data Historis":
    st.title("📋 Tabel Data Historis & Rekayasa Fitur")
    st.markdown("Rincian data transaksi bulanan kategori *Technology* beserta atribut *Lag*.")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df_monthly, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
