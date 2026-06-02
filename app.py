import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

try:
    import puresnmp
    SNMP_AVAILABLE = True
except Exception:
    SNMP_AVAILABLE = False

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Analisis Penggunaan Bandwidth",
    page_icon="📡",
    layout="wide",
)

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg, #f7f9ff 0%, #ffffff 50%, #fff5fa 100%);} 
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
    .hero {
        background: #ffffff;
        border: 1px solid #ececf7;
        padding: 28px 30px;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(31, 41, 55, 0.08);
        margin-bottom: 20px;
    }
    .main-title {
        font-size: 48px;
        font-weight: 900;
        color: #111827;
        margin: 0;
        line-height: 1.08;
        letter-spacing: -0.8px;
    }
    .main-subtitle {
        font-size: 17px;
        color: #4b5563;
        margin-top: 10px;
        margin-bottom: 0px;
    }
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #ececf7;
        padding: 16px;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
    }
    .card {
        background:#ffffff;
        border:1px solid #ececf7;
        border-radius:22px;
        padding:18px 20px;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
        margin-bottom:14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <p class="main-title">AI Analisis Penggunaan Bandwidth</p>
        <p class="main-subtitle">
        Dashboard analisis bandwidth menggunakan AI untuk klasifikasi, monitoring real-time, deteksi anomali,
        status jaringan, network mood, activity classification, dan rekomendasi otomatis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data(path="bandwidth_prediction_dataset.csv"):
    return pd.read_csv(path).dropna().reset_index(drop=True)

try:
    data = load_data()
except FileNotFoundError:
    st.error("File `bandwidth_prediction_dataset.csv` belum ada di folder yang sama dengan `app.py`.")
    st.stop()

fitur = ["throughput", "jitter", "actual_bandwidth"]
missing_cols = [col for col in fitur if col not in data.columns]
if missing_cols:
    st.error("Dataset wajib memiliki kolom: throughput, jitter, actual_bandwidth")
    st.stop()

# Kolom tambahan agar dashboard tetap berjalan walaupun dataset sederhana
if "rtt" not in data.columns:
    data["rtt"] = data.get("packet_delay", data["jitter"] * 2)
if "packet_loss_ratio" not in data.columns:
    data["packet_loss_ratio"] = data.get("error_rate", 0)
if "hour" not in data.columns:
    data["hour"] = np.arange(len(data)) % 24

# =========================================================
# TRAINING MODEL
# =========================================================
@st.cache_resource
def train_models(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[fitur])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    mean_bandwidth = pd.Series(df["actual_bandwidth"]).groupby(clusters).mean().sort_values()
    label_cluster = {
        mean_bandwidth.index[0]: "Rendah",
        mean_bandwidth.index[1]: "Sedang",
        mean_bandwidth.index[2]: "Tinggi",
    }
    sil_score = silhouette_score(X_scaled, clusters)

    anomaly_features = ["actual_bandwidth", "throughput", "jitter", "rtt", "packet_loss_ratio"]
    anomaly_model = IsolationForest(contamination=0.08, random_state=42)
    anomaly_model.fit(df[anomaly_features])

    return scaler, kmeans, label_cluster, sil_score, anomaly_model

scaler, kmeans, label_cluster, score, anomaly_model = train_models(data)
data["Cluster"] = kmeans.predict(scaler.transform(data[fitur]))
data["Kategori_Bandwidth"] = data["Cluster"].map(label_cluster)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def classify_activity(download_ratio, upload_ratio, latency, jitter, packet_loss):
    if download_ratio > 0.78:
        return "Download"
    if download_ratio > 0.60 and latency < 100 and jitter < 45:
        return "Streaming"
    if upload_ratio > 0.40 and latency < 140 and jitter < 55:
        return "Video Meeting"
    if latency < 65 and jitter < 28 and packet_loss < 1.8:
        return "Gaming"
    return "Browsing"


def health_score(latency, jitter, packet_loss, usage_ratio):
    nilai = 100
    nilai -= min(latency / 3.0, 35)
    nilai -= min(jitter / 2.0, 25)
    nilai -= min(packet_loss * 8.0, 25)
    nilai -= min(usage_ratio * 15.0, 15)
    return int(max(0, min(100, nilai)))


def network_status(score_val, usage_ratio):
    if score_val >= 75 and usage_ratio < 0.70:
        return "Stabil"
    if score_val >= 45:
        return "Padat"
    return "Overload"


def network_mood(latency, jitter, packet_loss, usage_ratio):
    buruk = 0
    buruk += latency > 120
    buruk += jitter > 50
    buruk += packet_loss > 3
    buruk += usage_ratio > 0.80
    if buruk == 0:
        return "Calm"
    if buruk == 1:
        return "Busy"
    if buruk == 2:
        return "Chaotic"
    return "Frustrated"


def recommendation(status, mood, activity, latency, jitter, packet_loss, usage_ratio):
    recs = []
    if activity in ["Download", "Streaming"] or usage_ratio > 0.78:
        recs.append("Kurangi download besar atau streaming HD saat jam ramai.")
    if latency > 120:
        recs.append("Latency tinggi: cek jumlah user aktif, posisi router, dan aplikasi berat.")
    if jitter > 50:
        recs.append("Jitter tinggi: prioritaskan meeting/game dan kurangi traffic background.")
    if packet_loss > 3:
        recs.append("Packet loss tinggi: cek kabel, interferensi Wi-Fi, atau restart router.")
    if status == "Stabil" and mood == "Calm":
        recs.append("Jaringan stabil dan aman untuk browsing, meeting, serta streaming normal.")
    if not recs:
        recs.append("Kondisi cukup aman. Tetap pantau jam sibuk dan penggunaan bandwidth.")
    return " ".join(recs[:3])


def ai_insight(df):
    if df.empty:
        return "Belum ada data monitoring."
    temp = df.copy()
    if "Jam" not in temp.columns:
        temp["Jam"] = pd.to_datetime(temp["Waktu"], errors="coerce").dt.hour.fillna(0).astype(int)
    value_col = "Throughput" if "Throughput" in temp.columns else "actual_bandwidth"
    busy_hour = int(temp.groupby("Jam")[value_col].mean().idxmax())
    dominant = temp["Aktivitas"].mode()[0] if "Aktivitas" in temp.columns and not temp["Aktivitas"].mode().empty else "Belum terdeteksi"
    avg_latency = temp["Latency"].mean() if "Latency" in temp.columns else temp.get("rtt", pd.Series([0])).mean()
    avg_loss = temp["Packet Loss"].mean() if "Packet Loss" in temp.columns else temp.get("packet_loss_ratio", pd.Series([0])).mean()
    avg_jitter = temp["Jitter"].mean() if "Jitter" in temp.columns else temp.get("jitter", pd.Series([0])).mean()
    kondisi = "cukup stabil" if avg_latency < 100 and avg_loss < 2 and avg_jitter < 40 else "perlu perhatian"
    return f"Jam tersibuk sekitar pukul {busy_hour}:00. Aktivitas dominan adalah {dominant}. Kondisi jaringan {kondisi}."


def historical_dashboard_data(n=100):
    hist = data.tail(n).copy().reset_index(drop=True)
    max_bw = max(hist["actual_bandwidth"].quantile(0.95), 1)
    df = pd.DataFrame({
        "Waktu": pd.date_range(end=datetime.now(), periods=len(hist), freq="min"),
        "Jam": hist["hour"].astype(int).values,
        "Download": hist["actual_bandwidth"].values,
        "Upload": hist["throughput"].values * 0.35,
        "Throughput": hist["throughput"].values,
        "Latency": hist["rtt"].values,
        "Jitter": hist["jitter"].values,
        "Packet Loss": hist["packet_loss_ratio"].values,
    })
    df["Usage"] = ((df["Download"] + df["Upload"]) / (max_bw * 1.3) * 100).clip(0, 100)
    df["Aktivitas"] = df.apply(lambda r: classify_activity(r["Download"] / max_bw, r["Upload"] / max_bw, r["Latency"], r["Jitter"], r["Packet Loss"]), axis=1)
    df["Health Score"] = df.apply(lambda r: health_score(r["Latency"], r["Jitter"], r["Packet Loss"], r["Usage"] / 100), axis=1)
    df["Status"] = df.apply(lambda r: network_status(r["Health Score"], r["Usage"] / 100), axis=1)
    df["Mood"] = df.apply(lambda r: network_mood(r["Latency"], r["Jitter"], r["Packet Loss"], r["Usage"] / 100), axis=1)
    df["Rekomendasi"] = df.apply(lambda r: recommendation(r["Status"], r["Mood"], r["Aktivitas"], r["Latency"], r["Jitter"], r["Packet Loss"], r["Usage"] / 100), axis=1)
    return df


def generate_realtime_row(download=None, upload=None, source="Simulasi"):
    max_bw = max(data["actual_bandwidth"].quantile(0.95), 1)
    download = float(download if download is not None else np.random.uniform(0.15, 0.95) * max_bw)
    upload = float(upload if upload is not None else np.random.uniform(0.05, 0.45) * max_bw)
    latency = float(np.random.uniform(25, 185))
    jitter = float(np.random.uniform(5, 85))
    packet_loss = float(np.random.uniform(0, 6))
    usage_ratio = min(1.0, (download + upload) / (max_bw * 1.3))
    activity = classify_activity(download / max_bw, upload / max_bw, latency, jitter, packet_loss)
    hscore = health_score(latency, jitter, packet_loss, usage_ratio)
    status = network_status(hscore, usage_ratio)
    mood = network_mood(latency, jitter, packet_loss, usage_ratio)
    rec = recommendation(status, mood, activity, latency, jitter, packet_loss, usage_ratio)
    return {
        "Waktu": datetime.now(),
        "Jam": datetime.now().hour,
        "Download": round(download, 2),
        "Upload": round(upload, 2),
        "Throughput": round(download + upload, 2),
        "Latency": round(latency, 2),
        "Jitter": round(jitter, 2),
        "Packet Loss": round(packet_loss, 2),
        "Usage": round(usage_ratio * 100, 2),
        "Health Score": hscore,
        "Status": status,
        "Mood": mood,
        "Aktivitas": activity,
        "Rekomendasi": rec,
        "Sumber": source,
    }


def get_snmp_octets(host, community, oid_in, oid_out, port):
    if not SNMP_AVAILABLE:
        return None, None
    try:
        val_in = puresnmp.get(host, community, oid_in, port=int(port), timeout=3)
        val_out = puresnmp.get(host, community, oid_out, port=int(port), timeout=3)
        return int(val_in), int(val_out)
    except Exception:
        return None, None

# =========================================================
# SESSION STATE
# =========================================================
if "monitoring_aktif" not in st.session_state:
    st.session_state.monitoring_aktif = False
if "history_monitoring" not in st.session_state:
    st.session_state.history_monitoring = []
if "prev_in" not in st.session_state:
    st.session_state.prev_in = None
if "prev_out" not in st.session_state:
    st.session_state.prev_out = None
if "polling_count" not in st.session_state:
    st.session_state.polling_count = 0

# =========================================================
# SIDEBAR - HANYA 5 FITUR SESUAI PERMINTAAN
# =========================================================
st.sidebar.markdown("### 📡 Menu Dashboard")
menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Dashboard Utama",
        "Dataset Training",
        "Input Manual",
        "Upload Dataset Baru",
        "Monitoring Real-Time (SNMP)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Status Sistem")
st.sidebar.write("SNMP:", "Tersedia" if SNMP_AVAILABLE else "Belum terinstall")
st.sidebar.write("Model:", "K-Means + Isolation Forest")

if st.session_state.history_monitoring:
    df_dash = pd.DataFrame(st.session_state.history_monitoring)
else:
    df_dash = historical_dashboard_data(100)

# =========================================================
# 1. DASHBOARD UTAMA
# =========================================================
if menu == "Dashboard Utama":
    st.subheader("Dashboard Utama")
    latest = df_dash.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Download", f"{df_dash['Download'].sum():,.0f}")
    c2.metric("Total Upload", f"{df_dash['Upload'].sum():,.0f}")
    c3.metric("Throughput", f"{latest['Throughput']:,.0f}")
    c4.metric("Health Score", f"{int(latest['Health Score'])}/100")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Latency / Ping", f"{latest['Latency']:.2f} ms")
    c6.metric("Jitter", f"{latest['Jitter']:.2f} ms")
    c7.metric("Packet Loss", f"{latest['Packet Loss']:.2f}%")
    c8.metric("Status Jaringan", latest["Status"])

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.markdown("### Grafik Download / Upload")
        st.plotly_chart(px.line(df_dash, x="Waktu", y=["Download", "Upload"], markers=True), use_container_width=True)
        st.markdown("### Grafik Latency")
        st.plotly_chart(px.line(df_dash, x="Waktu", y="Latency", markers=True), use_container_width=True)

    with col_right:
        st.markdown("### AI Insight Panel")
        st.markdown(f"<div class='card'>{ai_insight(df_dash)}</div>", unsafe_allow_html=True)
        st.metric("Network Mood", latest["Mood"])
        st.metric("Aktivitas Dominan", latest["Aktivitas"])
        st.metric("Rekomendasi", latest["Rekomendasi"][:60] + "...")
        st.plotly_chart(px.pie(df_dash, names="Aktivitas", title="Pie Chart Aktivitas"), use_container_width=True)

    st.markdown("### Heatmap Jam Sibuk")
    per_jam = df_dash.groupby("Jam", as_index=False)[["Download", "Upload", "Throughput"]].mean()
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(px.bar(per_jam, x="Jam", y="Throughput", title="Penggunaan Bandwidth per Jam"), use_container_width=True)
    with h2:
        fig_heat = go.Figure(data=go.Heatmap(z=per_jam[["Throughput"]].T.values, x=per_jam["Jam"], y=["Bandwidth Usage"]))
        fig_heat.update_layout(title="Heatmap Jam Sibuk")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Anomaly Detection")
    df_anom = df_dash.copy()
    spike_limit = df_anom["Throughput"].mean() + 2 * df_anom["Throughput"].std()
    df_anom["Anomaly"] = np.where(
        (df_anom["Throughput"] > spike_limit) | (df_anom["Latency"] > 150) | (df_anom["Packet Loss"] > 4),
        "Warning",
        "Normal",
    )
    warnings = (df_anom["Anomaly"] == "Warning").sum()
    if warnings:
        st.warning(f"Terdeteksi {warnings} potensi gangguan: traffic spike, latency tinggi, atau packet loss tinggi.")
    else:
        st.success("Tidak ada anomali besar pada data terakhir.")
    st.dataframe(df_anom.tail(15), use_container_width=True)

# =========================================================
# 2. DATASET TRAINING
# =========================================================
elif menu == "Dataset Training":
    st.subheader("Dataset Training")
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", data.shape[0])
    col2.metric("Jumlah Kolom", data.shape[1])
    col3.metric("Silhouette Score", round(score, 4))

    st.markdown("### Data Awal")
    st.dataframe(data.head(20), use_container_width=True)

    st.markdown("### Hasil Clustering")
    st.dataframe(data["Kategori_Bandwidth"].value_counts(), use_container_width=True)

    st.markdown("### Rata-rata Fitur Tiap Kategori")
    st.dataframe(data.groupby("Kategori_Bandwidth")[fitur].mean(), use_container_width=True)

    st.markdown("### Visualisasi Clustering")
    st.plotly_chart(px.scatter(data, x="throughput", y="actual_bandwidth", color="Kategori_Bandwidth"), use_container_width=True)

# =========================================================
# 3. INPUT MANUAL
# =========================================================
elif menu == "Input Manual":
    st.subheader("Analisis Bandwidth dengan Input Manual")
    col1, col2, col3 = st.columns(3)
    with col1:
        throughput = st.number_input("Throughput", min_value=0.0, value=100.0)
        latency = st.number_input("Latency/Ping (ms)", min_value=0.0, value=50.0)
    with col2:
        jitter = st.number_input("Jitter (ms)", min_value=0.0, value=10.0)
        packet_loss = st.number_input("Packet Loss (%)", min_value=0.0, value=1.0)
    with col3:
        actual_bandwidth = st.number_input("Download / Actual Bandwidth", min_value=0.0, value=300.0)
        upload = st.number_input("Upload Speed", min_value=0.0, value=50.0)

    if st.button("Analisis"):
        data_baru = pd.DataFrame([[throughput, jitter, actual_bandwidth]], columns=fitur)
        cluster = kmeans.predict(scaler.transform(data_baru))[0]
        kategori = label_cluster[cluster]

        max_bw = max(data["actual_bandwidth"].quantile(0.95), 1)
        usage_ratio = min(1.0, (actual_bandwidth + upload) / (max_bw * 1.3))
        activity = classify_activity(actual_bandwidth / max_bw, upload / max_bw, latency, jitter, packet_loss)
        hscore = health_score(latency, jitter, packet_loss, usage_ratio)
        status = network_status(hscore, usage_ratio)
        mood = network_mood(latency, jitter, packet_loss, usage_ratio)
        rec = recommendation(status, mood, activity, latency, jitter, packet_loss, usage_ratio)

        st.success("Hasil Analisis Berhasil")
        a, b, c, d = st.columns(4)
        a.metric("Kategori Bandwidth", kategori)
        b.metric("Aktivitas", activity)
        c.metric("Mood", mood)
        d.metric("Health Score", f"{hscore}/100")
        st.info(f"Status jaringan: {status}. Rekomendasi: {rec}")

# =========================================================
# 4. UPLOAD DATASET BARU
# =========================================================
elif menu == "Upload Dataset Baru":
    st.subheader("Analisis Bandwidth dengan Upload Dataset CSV")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        data_uji = pd.read_csv(uploaded_file).dropna().reset_index(drop=True)
        st.markdown("### Dataset yang Diupload")
        st.dataframe(data_uji.head(), use_container_width=True)

        if all(kolom in data_uji.columns for kolom in fitur):
            if "rtt" not in data_uji.columns:
                data_uji["rtt"] = data_uji.get("packet_delay", data_uji["jitter"] * 2)
            if "packet_loss_ratio" not in data_uji.columns:
                data_uji["packet_loss_ratio"] = data_uji.get("error_rate", 0)
            if "hour" not in data_uji.columns:
                data_uji["hour"] = np.arange(len(data_uji)) % 24

            data_uji["Cluster"] = kmeans.predict(scaler.transform(data_uji[fitur]))
            data_uji["Kategori_Bandwidth"] = data_uji["Cluster"].map(label_cluster)

            max_bw = max(data_uji["actual_bandwidth"].quantile(0.95), 1)
            data_uji["Aktivitas"] = data_uji.apply(
                lambda r: classify_activity(r["actual_bandwidth"] / max_bw, r["throughput"] / max_bw, r["rtt"], r["jitter"], r["packet_loss_ratio"]),
                axis=1,
            )
            data_uji["Anomaly"] = anomaly_model.predict(data_uji[["actual_bandwidth", "throughput", "jitter", "rtt", "packet_loss_ratio"]])
            data_uji["Anomaly"] = data_uji["Anomaly"].map({1: "Normal", -1: "Anomaly"})

            st.markdown("### Hasil Analisis Dataset Baru")
            st.dataframe(data_uji, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(data_uji, names="Aktivitas", title="Aktivitas Dominan"), use_container_width=True)
            with c2:
                st.plotly_chart(px.histogram(data_uji, x="Kategori_Bandwidth", title="Kategori Bandwidth"), use_container_width=True)

            st.download_button(
                "Download Hasil Analisis",
                data_uji.to_csv(index=False).encode("utf-8"),
                "hasil_analisis_dataset_baru.csv",
                "text/csv",
            )
        else:
            st.error("File CSV harus memiliki kolom: throughput, jitter, actual_bandwidth")

# =========================================================
# 5. MONITORING REAL-TIME SNMP
# =========================================================
elif menu == "Monitoring Real-Time (SNMP)":
    st.subheader("Monitoring Bandwidth Real-Time via SNMP")

    if not SNMP_AVAILABLE:
        st.info("Library `puresnmp` belum terinstall. Mode simulasi tetap bisa digunakan. Install dengan: `pip install puresnmp`")

    with st.expander("Konfigurasi SNMP", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            snmp_host = st.text_input("IP Perangkat", value="192.168.1.1")
            snmp_community = st.text_input("Community String", value="public")
            snmp_port = st.number_input("Port SNMP", value=161, min_value=1, max_value=65535)
        with col2:
            oid_in = st.text_input("OID In (ifInOctets)", value="1.3.6.1.2.1.2.2.1.10.1")
            oid_out = st.text_input("OID Out (ifOutOctets)", value="1.3.6.1.2.1.2.2.1.16.1")
            interval = st.slider("Interval Polling (detik)", 2, 30, 5)

    use_simulasi = st.checkbox("Gunakan Data Simulasi", value=not SNMP_AVAILABLE)

    c1, c2, c3 = st.columns(3)
    if c1.button("▶ Mulai Monitoring", disabled=st.session_state.monitoring_aktif):
        st.session_state.monitoring_aktif = True
        st.session_state.history_monitoring = []
        st.session_state.prev_in = None
        st.session_state.prev_out = None
        st.session_state.polling_count = 0
        st.rerun()

    if c2.button("⏹ Stop Monitoring", disabled=not st.session_state.monitoring_aktif):
        st.session_state.monitoring_aktif = False
        st.rerun()

    if c3.button("🧹 Hapus Riwayat"):
        st.session_state.history_monitoring = []
        st.rerun()

    if st.session_state.monitoring_aktif:
        st.session_state.polling_count += 1
        row = None

        if use_simulasi:
            row = generate_realtime_row(source="Simulasi")
        else:
            in_bytes, out_bytes = get_snmp_octets(snmp_host, snmp_community, oid_in, oid_out, snmp_port)
            if in_bytes is None or out_bytes is None:
                st.error("Gagal mengambil data SNMP. Periksa IP, community string, OID, dan status SNMP perangkat.")
            elif st.session_state.prev_in is None:
                st.session_state.prev_in = in_bytes
                st.session_state.prev_out = out_bytes
                st.info("Inisialisasi SNMP berhasil. Menunggu polling berikutnya.")
            else:
                download = max(0, in_bytes - st.session_state.prev_in) * 8 / 1000 / interval
                upload = max(0, out_bytes - st.session_state.prev_out) * 8 / 1000 / interval
                st.session_state.prev_in = in_bytes
                st.session_state.prev_out = out_bytes
                row = generate_realtime_row(download=download, upload=upload, source="SNMP")

        if row:
            st.session_state.history_monitoring.append(row)
            st.success(f"Polling #{st.session_state.polling_count} | Status: {row['Status']} | Mood: {row['Mood']} | Aktivitas: {row['Aktivitas']}")

        time.sleep(interval)
        st.rerun()

    if st.session_state.history_monitoring:
        df_hist = pd.DataFrame(st.session_state.history_monitoring)
        latest = df_hist.iloc[-1]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Download", f"{latest['Download']:,.2f}")
        m2.metric("Upload", f"{latest['Upload']:,.2f}")
        m3.metric("Health Score", f"{latest['Health Score']}/100")
        m4.metric("Status", latest["Status"])

        st.plotly_chart(px.line(df_hist, x="Waktu", y=["Download", "Upload"], markers=True), use_container_width=True)
        st.plotly_chart(px.line(df_hist, x="Waktu", y="Latency", markers=True), use_container_width=True)
        st.dataframe(df_hist, use_container_width=True)

        st.download_button(
            "Download History Monitoring",
            df_hist.to_csv(index=False).encode("utf-8"),
            "history_monitoring.csv",
            "text/csv",
        )
    else:
        st.info("Belum ada history monitoring.")
