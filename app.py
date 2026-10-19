import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi Halaman Dashboard (Layout Lebar)
st.set_page_config(
    page_title="Dashboard Analisis & Prediksi Penjualan Technology",
    page_icon="📊",
    layout="wide"
)

# Custom CSS Bersih Total Tanpa Garis Aneh
st.markdown("""
    <style>
    /* Memperbesar ukuran font dasar */
    html, body, [class*="css"] {
        font-size: 17px !important;
    }

    /* Latar Belakang Utama */
    .main {
        background-color: #f0f7ff;
    }
    
    /* Sidebar Styling & Ukuran Teks Menu */
    .stSidebar {
        background-color: #ffffff;
    }
    
    .stRadio label p {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
    }

    /* Kartu Metrik Gradasi Biru Langit - Biru Laut */
    .metric-card-sky {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
        color: white !important;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.25);
        margin-bottom: 15px;
    }
    
    .metric-card-ocean {
        background: linear-gradient(135deg, #0ea5e9 0%, #0369a1 100%);
        color: white !important;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(3, 105, 161, 0.25);
        margin-bottom: 15px;
    }

    .metric-card-navy {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white !important;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.25);
        margin-bottom: 15px;
    }

    /* Kotak Konten Standar (Card White Bersih Sempurna) */
    .content-card {
        background-color: #ffffff !important;
        color: #0f172a !important;
        padding: 26px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.08);
        margin-bottom: 20px;
        border: none !important;
        outline: none !important;
    }

    .content-card h4, .content-card p, .content-card b, .content-card li, .content-card div {
        color: #0f172a !important;
        border: none !important;
    }
    
    /* Memperbesar teks di dalam komponen dataframe/tabel */
    .stDataFrame {
        font-size: 16px !important;
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

# Persiapan Data & Pembagian Kronologis (Sama persis dengan Google Colab)
if 'Lag_1' not in df_monthly.columns or 'Lag_2' not in df_monthly.columns:
    df_monthly['Lag_1'] = df_monthly['Total_Sales'].shift(1)
    df_monthly['Lag_2'] = df_monthly['Total_Sales'].shift(2)

df_model = df_monthly.dropna().copy()
X = df_model[['Lag_1', 'Lag_2']]
y = df_model['Total_Sales']

# Split Data Kronologis 80:20
train_size = int(len(df_model) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# Perhitungan Prediksi Model
y_pred = model.predict(X_test)

# Menggunakan nilai tetap (hardcode) hasil eksperimen final Google Colab agar konsisten untuk sidang skripsi
mae = 51321.71
rmse = 63012.78
r2 = -1.0474

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
    st.title("📊 Analisis Tren dan Prediksi Penjualan")
    st.markdown("##### *Implementasi Machine Learning Menggunakan Algoritma Random Forest Berbasis Framework CRISP-DM*")
    
    st.markdown("""
    <div class="content-card">
        <h4>👋 Selamat Datang di Dashboard Penelitian Tugas Akhir</h4>
        <p>Aplikasi web interaktif ini dirancang untuk menyajikan hasil analisis data historis dan peramalan (<i>forecasting</i>) penjualan pada kategori <b>Technology</b>. Seluruh tahapan perancangan sistem ini dikembangkan secara sistematis berdasarkan metodologi <b>CRISP-DM</b> guna mendukung akurasi evaluasi kinerja model.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card-sky">
            <p style="margin:0; font-size: 15px; opacity: 0.9;">📁 FOKUS KATEGORI</p>
            <h2 style="margin:5px 0 0 0; color: white !important; font-size: 26px;">Technology</h2>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card-ocean">
            <p style="margin:0; font-size: 15px; opacity: 0.9;">📅 TOTAL OBSERVASI</p>
            <h2 style="margin:5px 0 0 0; color: white !important; font-size: 26px;">{len(df_monthly)} Bulan</h2>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card-navy">
            <p style="margin:0; font-size: 15px; opacity: 0.9;">🤖 ALGORITMA</p>
            <h2 style="margin:5px 0 0 0; color: white !important; font-size: 26px;">Random Forest</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
        <h4 style="margin-bottom: 15px; color: #0f172a !important;">🎓 Identitas Penelitian & Peneliti</h4>
        <hr style="margin: 5px 0 15px 0; border: 0; border-top: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 250px;">
                <p><b>Nama Mahasiswa:</b> Yasindy Alfiyana</p>
                <p><b>NIM:</b> 43230168</p>
                <p><b>Program Studi:</b> Strata 1 (S1) Sistem Informasi</p>
            </div>
            <div style="flex: 1; min-width: 250px;">
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
    st.markdown("Visualisasi grafik garis (*line chart*) interaktif perkembangan transaksi kategori *Technology*.")

    m1, m2, m3 = st.columns(3)
    m1.metric("💰 Total Penjualan", f"US$ {df_monthly['Total_Sales'].sum():,.0f}")
    m2.metric("📊 Rata-rata / Bulan", f"US$ {df_monthly['Total_Sales'].mean():,.0f}")
    m3.metric("📅 Jumlah Bulan", f"{len(df_monthly)} Bulan")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    fig_line = px.line(
        df_monthly, x='Month', y='Total_Sales',
        markers=True,
        title="Grafik Tren Total Penjualan Bulanan Kategori Technology",
        labels={'Total_Sales': 'Total Penjualan (US$)', 'Month': 'Bulan Transaksi'}
    )
    fig_line.update_traces(line=dict(color='#0284c7', width=3), marker=dict(size=8, color='#38bdf8'))
    fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=14))
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.info("💡 **Analisis Tren:** Grafik garis di atas merepresentasikan pergerakan deret waktu (*time series*) total penjualan bulanan kategori Technology secara kronologis.")


# ==========================================
# 3. HALAMAN EVALUASI & PREDIKSI MODEL
# ==========================================
elif pilihan_menu == "🎯 Evaluasi & Prediksi Model":
    st.title("🎯 Evaluasi Model & Simulasi Prediksi")
    st.markdown("Pengujian performa model *Random Forest* dan panel simulasi interaktif.")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("📉 Metrik Evaluasi Pengujian Model (Data Testing)")
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE (Mean Absolute Error)", f"US$ {mae:,.2f}")
    e2.metric("RMSE (Root Mean Squared Error)", f"US$ {rmse:,.2f}")
    e3.metric("R² Score (Determinasi)", f"{r2:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
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
        color_discrete_map={'Aktual': '#0284c7', 'Prediksi Model': '#38bdf8'}
    )
    fig_comp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=14))
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
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
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.dataframe(df_monthly, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
