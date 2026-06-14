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


# =========================================================
# SNMP CONFIG
# =========================================================
# SNMP tidak di-import di awal supaya dashboard tetap bisa jalan
# walaupun library SNMP bermasalah di Streamlit Cloud.
SNMP_AVAILABLE = True
SNMP_ERROR_MESSAGE = ""


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
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.22), transparent 32%),
            linear-gradient(135deg, #eaf3ff 0%, #f8fbff 48%, #dbeafe 100%);
        color: #0f172a;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    .hero {
        background: linear-gradient(135deg, #0f4c81 0%, #1d72d2 48%, #38bdf8 100%);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 30px 32px;
        border-radius: 26px;
        box-shadow: 0 18px 45px rgba(30, 64, 175, 0.28);
        margin-bottom: 22px;
    }

    .main-title {
        font-size: 46px;
        font-weight: 900;
        color: #ffffff !important;
        margin: 0;
        line-height: 1.08;
        letter-spacing: -0.8px;
        text-shadow: 0 2px 10px rgba(15,23,42,0.22);
    }

    .main-subtitle {
        font-size: 17px;
        color: #e0f2fe !important;
        margin-top: 12px;
        margin-bottom: 0px;
        line-height: 1.55;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f4c81 0%, #1e3a8a 100%);
        border-right: 1px solid rgba(255,255,255,0.14);
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(37,99,235,0.20);
        padding: 18px;
        border-radius: 20px;
        box-shadow: 0 10px 28px rgba(30, 64, 175, 0.12);
    }

    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #1e3a8a !important;
        font-weight: 800 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 900 !important;
    }

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #000000 !important;
        font-weight: 900 !important;
    }

    .card {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(37,99,235,0.20);
        border-radius: 22px;
        padding: 18px 20px;
        box-shadow: 0 10px 28px rgba(30, 64, 175, 0.12);
        margin-bottom: 14px;
        color: #0f172a !important;
        line-height: 1.6;
        font-weight: 600;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8, #0284c7);
        color: white !important;
        border: none;
        border-radius: 14px;
        padding: 0.62rem 1rem;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(37,99,235,0.25);
    }

    .stButton>button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(37,99,235,0.16);
    }

    .stAlert {
        border-radius: 16px;
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
    x_scaled = scaler.fit_transform(df[fitur])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(x_scaled)

    mean_bandwidth = pd.Series(df["actual_bandwidth"]).groupby(clusters).mean().sort_values()

    label_cluster = {
        mean_bandwidth.index[0]: "Rendah",
        mean_bandwidth.index[1]: "Sedang",
        mean_bandwidth.index[2]: "Tinggi",
    }

    sil_score = silhouette_score(x_scaled, clusters)

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
def format_speed(mbps):
    mbps = float(mbps)

    if mbps >= 1:
        return f"{mbps:,.2f} Mbps"

    if mbps >= 0.001:
        return f"{mbps * 1000:,.2f} Kbps"

    return f"{mbps * 1_000_000:,.0f} bps"


def classify_activity(download_ratio, upload_ratio, latency, jitter, packet_loss):
    download_ratio = float(download_ratio)
    upload_ratio = float(upload_ratio)
    total_ratio = download_ratio + upload_ratio

    if total_ratio < 0.05:
        return "Browsing"

    if download_ratio > 0.70 and total_ratio >= 0.10:
        return "Download"

    if download_ratio > 0.45 and latency < 120 and jitter < 35 and packet_loss < 2:
        return "Streaming"

    if upload_ratio > 0.30 and latency < 150 and jitter < 45 and packet_loss < 2.5:
        return "Video Meeting"

    if latency < 60 and jitter < 15 and packet_loss < 1.0 and total_ratio >= 0.12:
        return "Gaming"

    return "Browsing"


def health_score(latency, jitter, packet_loss, usage_ratio):
    nilai = 100
    nilai -= min(latency / 6.0, 20)
    nilai -= min(jitter / 4.0, 15)
    nilai -= min(packet_loss * 5.0, 20)
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

    dominant = (
        temp["Aktivitas"].mode()[0]
        if "Aktivitas" in temp.columns and not temp["Aktivitas"].mode().empty
        else "Belum terdeteksi"
    )

    avg_latency = temp["Latency"].mean() if "Latency" in temp.columns else temp.get("rtt", pd.Series([0])).mean()
    avg_loss = temp["Packet Loss"].mean() if "Packet Loss" in temp.columns else temp.get("packet_loss_ratio", pd.Series([0])).mean()
    avg_jitter = temp["Jitter"].mean() if "Jitter" in temp.columns else temp.get("jitter", pd.Series([0])).mean()
    avg_throughput = temp["Throughput"].mean() if "Throughput" in temp.columns else temp.get("actual_bandwidth", pd.Series([0])).mean()

    if avg_latency < 80 and avg_loss < 1.5 and avg_jitter < 25:
        kondisi = "cukup stabil"
    elif avg_latency < 140 and avg_loss < 3 and avg_jitter < 50:
        kondisi = "cukup padat, tetapi masih bisa digunakan"
    else:
        kondisi = "perlu perhatian"

    if len(temp) < 20:
        return (
            f"Data monitoring masih sedikit ({len(temp)} sampel). "
            f"Belum cukup untuk menentukan pola penggunaan jaringan. "
            f"Throughput rata-rata {format_speed(avg_throughput)}."
        )

    if temp["Jam"].nunique() < 2:
        return (
            f"Data monitoring sudah memiliki {len(temp)} sampel, tetapi masih berada pada jam yang sama. "
            f"Aktivitas dominan adalah {dominant}. "
            f"Throughput rata-rata {format_speed(avg_throughput)} dan kondisi jaringan {kondisi}."
        )

    value_col = "Throughput" if "Throughput" in temp.columns else "actual_bandwidth"
    busy_hour = int(temp.groupby("Jam")[value_col].mean().idxmax())

    return (
        f"Jam dengan penggunaan tertinggi pada data saat ini sekitar pukul {busy_hour}:00. "
        f"Aktivitas dominan adalah {dominant}. Kondisi jaringan {kondisi}."
    )


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

    df["Aktivitas"] = df.apply(
        lambda r: classify_activity(
            r["Download"] / max_bw,
            r["Upload"] / max_bw,
            r["Latency"],
            r["Jitter"],
            r["Packet Loss"],
        ),
        axis=1,
    )

    df["Health Score"] = df.apply(
        lambda r: health_score(
            r["Latency"],
            r["Jitter"],
            r["Packet Loss"],
            r["Usage"] / 100,
        ),
        axis=1,
    )

    df["Status"] = df.apply(
        lambda r: network_status(
            r["Health Score"],
            r["Usage"] / 100,
        ),
        axis=1,
    )

    df["Mood"] = df.apply(
        lambda r: network_mood(
            r["Latency"],
            r["Jitter"],
            r["Packet Loss"],
            r["Usage"] / 100,
        ),
        axis=1,
    )

    df["Rekomendasi"] = df.apply(
        lambda r: recommendation(
            r["Status"],
            r["Mood"],
            r["Aktivitas"],
            r["Latency"],
            r["Jitter"],
            r["Packet Loss"],
            r["Usage"] / 100,
        ),
        axis=1,
    )

    return df


def generate_realtime_row(download=None, upload=None, source="Simulasi", latency=None, jitter=None, packet_loss=None):
    max_bw = max(data["actual_bandwidth"].quantile(0.95), 1)

    if source == "Simulasi":
        download = float(download if download is not None else np.random.uniform(0.15, 0.95) * max_bw)
        upload = float(upload if upload is not None else np.random.uniform(0.05, 0.45) * max_bw)
        latency = float(latency if latency is not None else np.random.uniform(25, 120))
        jitter = float(jitter if jitter is not None else np.random.uniform(3, 35))
        packet_loss = float(packet_loss if packet_loss is not None else np.random.uniform(0, 2.5))
    else:
        download = float(download or 0.0)
        upload = float(upload or 0.0)

        throughput_now = download + upload

        latency = float(latency if latency is not None else 25 + min(throughput_now * 1.8, 60))
        jitter = float(jitter if jitter is not None else 3 + min(throughput_now * 0.9, 25))
        packet_loss = float(packet_loss if packet_loss is not None else min(max(throughput_now - 10, 0) * 0.04, 2.0))

    usage_ratio = min(1.0, (download + upload) / (max_bw * 1.3))

    activity = classify_activity(
        download / max_bw,
        upload / max_bw,
        latency,
        jitter,
        packet_loss,
    )

    if source != "Simulasi":
        throughput_now = float(download + upload)

        if throughput_now <= 0.0001:
            activity = "Idle"
        elif throughput_now < 0.03:
            activity = "Browsing"
        elif throughput_now < 0.50 and latency < 80 and jitter < 15 and packet_loss < 1:
            activity = "Gaming"
        elif upload > 0.5 and upload >= download * 0.8:
            activity = "Video Meeting"
        elif download > 1.0 and download >= upload * 3:
            activity = "Streaming"
        elif throughput_now >= 1.0 and latency < 60 and jitter < 15 and packet_loss < 1:
            activity = "Gaming"
        elif download > 2.0 and download >= upload * 2:
            activity = "Download"
        else:
            activity = "Browsing"

    throughput_now = float(download + upload)

    try:
        sample_ai = pd.DataFrame(
            [[throughput_now, float(jitter), float(download)]],
            columns=fitur,
        )

        cluster_ai = kmeans.predict(scaler.transform(sample_ai))[0]
        kategori_ai = label_cluster.get(cluster_ai, "Tidak diketahui")
    except Exception:
        kategori_ai = "Tidak diketahui"

    hscore = health_score(latency, jitter, packet_loss, usage_ratio)
    status = network_status(hscore, usage_ratio)
    mood = network_mood(latency, jitter, packet_loss, usage_ratio)
    rec = recommendation(status, mood, activity, latency, jitter, packet_loss, usage_ratio)

    return {
        "Waktu": datetime.now(),
        "Jam": datetime.now().hour,
        "Download": float(download),
        "Upload": float(upload),
        "Throughput": throughput_now,
        "Latency": round(latency, 2),
        "Jitter": round(jitter, 2),
        "Packet Loss": round(packet_loss, 2),
        "Usage": round(usage_ratio * 100, 2),
        "Kategori AI": kategori_ai,
        "Health Score": hscore,
        "Status": status,
        "Mood": mood,
        "Aktivitas": activity,
        "Rekomendasi": rec,
        "Sumber": source,
    }


# =========================================================
# SNMP FUNCTIONS
# =========================================================
def _normalize_snmp_value(value):
    """Ubah hasil SNMP menjadi integer."""
    try:
        return int(value)
    except Exception:
        return int(str(value))


def snmp_get_value(host, community, oid, port=161, timeout=2, retries=1):
    """
    Ambil 1 nilai SNMP dari MikroTik.

    Library pysnmp di-import di dalam fungsi agar dashboard tetap berjalan
    walaupun fitur SNMP tidak berhasil dipakai di Streamlit Cloud.
    """
    try:
        import asyncio
        from pysnmp.hlapi.asyncio import (
            SnmpEngine,
            CommunityData,
            UdpTransportTarget,
            ContextData,
            ObjectType,
            ObjectIdentity,
            get_cmd,
        )
    except Exception as e:
        return None, f"Library pysnmp belum siap: {e}"

    async def _get():
        try:
            target = await UdpTransportTarget.create(
                (host, int(port)),
                timeout=timeout,
                retries=retries,
            )

            result = await get_cmd(
                SnmpEngine(),
                CommunityData(community, mpModel=1),
                target,
                ContextData(),
                ObjectType(ObjectIdentity(oid)),
            )

            error_indication, error_status, error_index, var_binds = result

            if error_indication:
                return None, str(error_indication)

            if error_status:
                return None, str(error_status.prettyPrint())

            for _, value in var_binds:
                return _normalize_snmp_value(value), None

            return None, "Tidak ada data SNMP yang dikembalikan."

        except Exception as e:
            return None, str(e)

    try:
        return asyncio.run(_get())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_get())
        finally:
            loop.close()


def get_snmp_octets(host, community, oid_in, oid_out, port):
    """Ambil total byte masuk dan keluar dari interface MikroTik."""
    val_in, err_in = snmp_get_value(host, community, oid_in, port)

    if err_in:
        return None, None, err_in

    val_out, err_out = snmp_get_value(host, community, oid_out, port)

    if err_out:
        return None, None, err_out

    return val_in, val_out, None


def hitung_mbps(byte_now, byte_prev, interval_detik):
    """Hitung Mbps dari selisih byte SNMP."""
    if byte_now is None or byte_prev is None:
        return 0.0

    delta = max(0, int(byte_now) - int(byte_prev))

    return (delta * 8) / max(float(interval_detik), 1.0) / 1_000_000


def recent_nonzero_or_avg(df, col, window=6):
    """
    Ambil nilai terbaru yang tidak nol.
    Kalau semua nol, pakai rata-rata window.
    """
    if df is None or df.empty or col not in df.columns:
        return 0.0

    recent = pd.to_numeric(df[col].tail(window), errors="coerce").fillna(0)
    nonzero = recent[recent > 0]

    if not nonzero.empty:
        return float(nonzero.iloc[-1])

    return float(recent.mean())


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

if "prev_time" not in st.session_state:
    st.session_state.prev_time = None

if "polling_count" not in st.session_state:
    st.session_state.polling_count = 0


# =========================================================
# SIDEBAR
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
st.sidebar.success("SNMP: Mode siap diuji")
st.sidebar.caption("Tes koneksi SNMP ada di menu Monitoring Real-Time.")
st.sidebar.write("Model: K-Means + Isolation Forest")

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

    display_download = recent_nonzero_or_avg(df_dash, "Download")
    display_upload = recent_nonzero_or_avg(df_dash, "Upload")
    display_throughput = max(
        recent_nonzero_or_avg(df_dash, "Throughput"),
        display_download + display_upload,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Download Terbaru", format_speed(display_download))
    c2.metric("Upload Terbaru", format_speed(display_upload))
    c3.metric("Throughput", format_speed(display_throughput))
    c4.metric("Health Score", f"{int(latest['Health Score'])}/100")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Latency / Ping", f"{latest['Latency']:.2f} ms")
    c6.metric("Jitter", f"{latest['Jitter']:.2f} ms")
    c7.metric("Packet Loss", f"{latest['Packet Loss']:.2f}%")
    c8.metric("Status Jaringan", latest["Status"])

    if "Kategori AI" in df_dash.columns:
        st.metric("Kategori AI Bandwidth", latest.get("Kategori AI", "Tidak diketahui"))

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("### Grafik Download / Upload")
        st.plotly_chart(
            px.line(df_dash, x="Waktu", y=["Download", "Upload"], markers=True),
            use_container_width=True,
        )

        st.markdown("### Grafik Latency")
        st.plotly_chart(
            px.line(df_dash, x="Waktu", y="Latency", markers=True),
            use_container_width=True,
        )

    with col_right:
        st.markdown("### AI Insight Panel")
        st.markdown(f"<div class='card'>{ai_insight(df_dash)}</div>", unsafe_allow_html=True)

        st.metric("Network Mood", latest["Mood"])
        st.metric("Aktivitas Dominan", latest["Aktivitas"])
        st.metric("Rekomendasi", latest["Rekomendasi"][:60] + "...")

        st.plotly_chart(
            px.pie(df_dash, names="Aktivitas", title="Pie Chart Aktivitas"),
            use_container_width=True,
        )

    st.markdown("### Heatmap Jam Sibuk")

    per_jam = df_dash.groupby("Jam", as_index=False)[["Download", "Upload", "Throughput"]].mean()

    h1, h2 = st.columns(2)

    with h1:
        st.plotly_chart(
            px.bar(per_jam, x="Jam", y="Throughput", title="Penggunaan Bandwidth per Jam"),
            use_container_width=True,
        )

    with h2:
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=per_jam[["Throughput"]].T.values,
                x=per_jam["Jam"],
                y=["Bandwidth Usage"],
            )
        )

        fig_heat.update_layout(title="Heatmap Jam Sibuk")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Anomaly Detection")

    df_anom = df_dash.copy()
    spike_limit = df_anom["Throughput"].mean() + 2 * df_anom["Throughput"].std()

    df_anom["Anomaly"] = np.where(
        (df_anom["Throughput"] > spike_limit)
        | (df_anom["Latency"] > 150)
        | (df_anom["Packet Loss"] > 4),
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
    st.plotly_chart(
        px.scatter(
            data,
            x="throughput",
            y="actual_bandwidth",
            color="Kategori_Bandwidth",
        ),
        use_container_width=True,
    )


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

        activity = classify_activity(
            actual_bandwidth / max_bw,
            upload / max_bw,
            latency,
            jitter,
            packet_loss,
        )

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
                lambda r: classify_activity(
                    r["actual_bandwidth"] / max_bw,
                    r["throughput"] / max_bw,
                    r["rtt"],
                    r["jitter"],
                    r["packet_loss_ratio"],
                ),
                axis=1,
            )

            data_uji["Anomaly"] = anomaly_model.predict(
                data_uji[["actual_bandwidth", "throughput", "jitter", "rtt", "packet_loss_ratio"]]
            )

            data_uji["Anomaly"] = data_uji["Anomaly"].map({1: "Normal", -1: "Anomaly"})

            st.markdown("### Hasil Analisis Dataset Baru")
            st.dataframe(data_uji, use_container_width=True)

            c1, c2 = st.columns(2)

            with c1:
                st.plotly_chart(
                    px.pie(data_uji, names="Aktivitas", title="Aktivitas Dominan"),
                    use_container_width=True,
                )

            with c2:
                st.plotly_chart(
                    px.histogram(data_uji, x="Kategori_Bandwidth", title="Kategori Bandwidth"),
                    use_container_width=True,
                )

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

    st.info(
        "SNMP akan diuji saat tombol `Tes Baca SNMP Sekarang` ditekan. "
        "Kalau pakai MikroTik lokal dengan IP seperti 192.168.88.1, sebaiknya jalankan aplikasi dari laptop "
        "yang satu jaringan dengan MikroTik. Link Streamlit Cloud biasanya tidak bisa membaca IP lokal router."
    )

    with st.expander("Konfigurasi SNMP", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            snmp_host = st.text_input("IP Perangkat MikroTik", value="192.168.88.1")
            snmp_community = st.text_input("Community String", value="public")
            snmp_port = st.number_input("Port SNMP", value=161, min_value=1, max_value=65535)

        with col2:
            interface_index = st.number_input("Nomor Interface yang Dipantau", value=2, min_value=1, max_value=50)
            oid_in = st.text_input(
                "OID In / RX / ifInOctets",
                value=f"1.3.6.1.2.1.2.2.1.10.{interface_index}",
            )
            oid_out = st.text_input(
                "OID Out / TX / ifOutOctets",
                value=f"1.3.6.1.2.1.2.2.1.16.{interface_index}",
            )
            interval = st.slider("Interval Polling (detik)", 2, 30, 5)

        st.caption(
            "Interface index 1 biasanya ether1/WAN, index 2 biasanya ether2/LAN. "
            "Di MikroTik, aktifkan SNMP lewat IP > SNMP, lalu pastikan community sesuai."
        )

        if st.button("🔎 Tes Baca SNMP Sekarang"):
            test_in_1, test_out_1, test_err_1 = get_snmp_octets(
                snmp_host,
                snmp_community,
                oid_in,
                oid_out,
                snmp_port,
            )

            if test_err_1:
                st.error(f"SNMP masih gagal: {test_err_1}")
            else:
                with st.spinner("SNMP berhasil membaca counter awal. Mengukur selisih 3 detik..."):
                    time.sleep(3)

                test_in_2, test_out_2, test_err_2 = get_snmp_octets(
                    snmp_host,
                    snmp_community,
                    oid_in,
                    oid_out,
                    snmp_port,
                )

                if test_err_2:
                    st.error(f"SNMP counter awal terbaca, tapi counter kedua gagal: {test_err_2}")
                else:
                    test_download = hitung_mbps(test_in_2, test_in_1, 3)
                    test_upload = hitung_mbps(test_out_2, test_out_1, 3)

                    st.success("SNMP berhasil membaca counter MikroTik dan menghitung trafik.")

                    t1, t2, t3, t4 = st.columns(4)
                    t1.metric("Raw In Awal", f"{test_in_1:,}")
                    t2.metric("Raw In Akhir", f"{test_in_2:,}")
                    t3.metric("Download Terukur", format_speed(test_download))
                    t4.metric("Upload Terukur", format_speed(test_upload))

                    st.caption(
                        "Kalau speed kecil, itu normal saat trafik ringan. "
                        "Coba buka YouTube/download kecil agar counter berubah."
                    )

    use_simulasi = st.checkbox("Gunakan Data Simulasi", value=False)

    c1, c2, c3 = st.columns(3)

    if c1.button("▶ Mulai Monitoring", disabled=st.session_state.monitoring_aktif):
        st.session_state.monitoring_aktif = True
        st.session_state.history_monitoring = []
        st.session_state.prev_in = None
        st.session_state.prev_out = None
        st.session_state.prev_time = None
        st.session_state.polling_count = 0
        st.rerun()

    if c2.button("⏹ Stop Monitoring", disabled=not st.session_state.monitoring_aktif):
        st.session_state.monitoring_aktif = False
        st.rerun()

    if c3.button("🧹 Hapus Riwayat"):
        st.session_state.history_monitoring = []
        st.session_state.prev_in = None
        st.session_state.prev_out = None
        st.session_state.prev_time = None
        st.session_state.polling_count = 0
        st.rerun()

    if st.session_state.monitoring_aktif:
        st.session_state.polling_count += 1

        row = None

        if use_simulasi:
            row = generate_realtime_row(source="Simulasi")

        else:
            in_bytes, out_bytes, err = get_snmp_octets(
                snmp_host,
                snmp_community,
                oid_in,
                oid_out,
                snmp_port,
            )

            if err is not None:
                st.error(
                    f"Gagal mengambil data SNMP dari {snmp_host}:{snmp_port}. Detail: {err}. "
                    "Periksa IP MikroTik, community string, OID interface, SNMP di MikroTik, dan koneksi jaringan."
                )

            elif st.session_state.prev_in is None:
                st.session_state.prev_in = int(in_bytes)
                st.session_state.prev_out = int(out_bytes)
                st.session_state.prev_time = time.time()
                st.info("Inisialisasi SNMP berhasil. Menunggu polling berikutnya.")

            else:
                now_time = time.time()
                elapsed = max(now_time - float(st.session_state.prev_time or now_time), 0.5)

                delta_in = max(0, int(in_bytes) - int(st.session_state.prev_in))
                delta_out = max(0, int(out_bytes) - int(st.session_state.prev_out))

                raw_in_mbps = hitung_mbps(in_bytes, st.session_state.prev_in, elapsed)
                raw_out_mbps = hitung_mbps(out_bytes, st.session_state.prev_out, elapsed)

                if int(interface_index) == 1:
                    download = raw_in_mbps
                    upload = raw_out_mbps
                else:
                    download = raw_out_mbps
                    upload = raw_in_mbps

                st.session_state.prev_in = int(in_bytes)
                st.session_state.prev_out = int(out_bytes)
                st.session_state.prev_time = now_time

                row = generate_realtime_row(
                    download=download,
                    upload=upload,
                    source="SNMP",
                    latency=25,
                    jitter=3,
                    packet_loss=0,
                )

                row["Raw In Mbps"] = raw_in_mbps
                row["Raw Out Mbps"] = raw_out_mbps
                row["Raw In"] = int(in_bytes)
                row["Raw Out"] = int(out_bytes)
                row["Delta In Byte"] = int(delta_in)
                row["Delta Out Byte"] = int(delta_out)
                row["Interval Detik"] = round(elapsed, 2)
                row["Download Display"] = format_speed(download)
                row["Upload Display"] = format_speed(upload)

        if row:
            st.session_state.history_monitoring.append(row)
            st.success(
                f"Polling #{st.session_state.polling_count} | "
                f"Status: {row['Status']} | Mood: {row['Mood']} | Aktivitas: {row['Aktivitas']}"
            )

        time.sleep(interval)
        st.rerun()

    if st.session_state.history_monitoring:
        df_hist = pd.DataFrame(st.session_state.history_monitoring)
        latest = df_hist.iloc[-1]

        display_download = recent_nonzero_or_avg(df_hist, "Download")
        display_upload = recent_nonzero_or_avg(df_hist, "Upload")
        display_throughput = max(
            recent_nonzero_or_avg(df_hist, "Throughput"),
            display_download + display_upload,
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Download", format_speed(display_download))
        m2.metric("Upload", format_speed(display_upload))
        m3.metric("Throughput", format_speed(display_throughput))
        m4.metric("Status", latest["Status"])
        m5.metric("Kategori AI", latest.get("Kategori AI", "Tidak diketahui"))

        fig_speed = px.line(df_hist.tail(30), x="Waktu", y=["Download", "Upload"], markers=True)
        fig_speed.update_layout(xaxis_title="Waktu", yaxis_title="Mbps")
        st.plotly_chart(fig_speed, use_container_width=True)

        fig_latency = px.line(df_hist.tail(30), x="Waktu", y="Latency", markers=True)
        fig_latency.update_layout(xaxis_title="Waktu", yaxis_title="ms")
        st.plotly_chart(fig_latency, use_container_width=True)

        st.dataframe(df_hist.tail(20), use_container_width=True, height=400)

        if "Delta In Byte" in df_hist.columns:
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Delta In Terakhir", f"{int(latest.get('Delta In Byte', 0)):,} byte")
            d2.metric("Delta Out Terakhir", f"{int(latest.get('Delta Out Byte', 0)):,} byte")
            d3.metric("Interval Aktual", f"{float(latest.get('Interval Detik', interval)):.2f} detik")
            d4.metric("RX/In SNMP", format_speed(latest.get("Raw In Mbps", 0)))
            d5.metric("TX/Out SNMP", format_speed(latest.get("Raw Out Mbps", 0)))

        st.caption(
            "Untuk interface LAN seperti ether2, Download dihitung dari TX/Out MikroTik ke laptop, "
            "sedangkan Upload dihitung dari RX/In laptop ke MikroTik."
        )

        st.download_button(
            "Download History Monitoring",
            df_hist.to_csv(index=False).encode("utf-8"),
            "history_monitoring.csv",
            "text/csv",
        )

    else:
        st.info(
            "Belum ada history monitoring. Klik `Tes Baca SNMP Sekarang` dulu, lalu klik `Mulai Monitoring`. "
            "Kalau tidak pakai MikroTik langsung, centang `Gunakan Data Simulasi`."
        )
