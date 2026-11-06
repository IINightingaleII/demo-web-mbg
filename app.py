import csv
import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from model_loader import USE_DUMMY, MODEL_DIR, load_predictor
from preprocess import preprocess_pair

TITLE = "Demo Analisis Sentimen & Deteksi Sarkasme Opini MBG"
DISCLAIMER = "Hasil merupakan prediksi model penelitian, bukan penilaian resmi terhadap program MBG."
DUMMY_DISCLAIMER = "Mode dummy: hasil simulasi deterministik, bukan keluaran checkpoint IndoBERTweet dan bukan penilaian resmi terhadap program MBG."
MAX_TEXT_CHARS = 10000
MAX_CSV_BYTES = 10 * 1024 * 1024

CSS = """
<style>
.stApp {font-family: 'Segoe UI', Arial, sans-serif;}
[data-testid="stMainBlockContainer"] {max-width: 70rem; padding-top: 2.5rem;}
h1 {font-family: Georgia, 'Times New Roman', serif; font-size: clamp(1.7rem, 3vw, 2.6rem) !important; line-height: 1.2;}
h2 {font-size: clamp(1.25rem, 2vw, 1.6rem) !important;}
p, label {line-height: 1.6;}
[data-testid="stMain"] :is(button, a[href], summary, select, label:has(input), [role="radio"]) {cursor: pointer;}
[data-testid="stMain"] button:disabled {cursor: not-allowed;}
[data-testid="stMain"] button {min-height: 44px;}
[data-testid="stMain"] :is(button, a[href], textarea, summary):focus-visible {outline: 3px solid #8A5B0A !important; outline-offset: 3px;}
[data-testid="stMain"] button:hover {text-decoration: underline;}
[data-testid="stMain"] button:active {transform: translateY(1px);}
[data-testid="stSidebar"] :is(button, a[href], label:has(input)) {cursor: pointer;}
[data-testid="stRadio"] label {min-height: 44px;}
[data-testid="stMetricValue"] {font-size: 1.7rem; white-space: normal;}
textarea {line-height: 1.6 !important;}
pre {white-space: pre-wrap; overflow-wrap: anywhere;}
@media (max-width: 850px) {
 [data-testid="stMainBlockContainer"] {padding: 2rem 1.2rem;}
 [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
 [data-testid="stColumn"] {min-width: min(100%, 20rem);}
}
@media (max-width: 600px) {
 [data-testid="stMainBlockContainer"] {padding: 1.5rem 1rem;}
 [data-testid="stHorizontalBlock"] {flex-direction: column;}
 [data-testid="stColumn"] {width: 100% !important; flex: 1 1 100% !important;}
}
@media (prefers-reduced-motion: reduce) {
 *, *::before, *::after {animation: none !important; transition: none !important; scroll-behavior: auto !important;}
}
</style>
"""


@st.cache_resource(show_spinner="Memuat prediktor...")
def get_model():
    return load_predictor()


def prepare_input(text, parent_text=None):
    pair = preprocess_pair(text, parent_text)
    shortened = []
    for key, value in pair.items():
        if len(value) > MAX_TEXT_CHARS:
            pair[key] = value[:MAX_TEXT_CHARS].rstrip()
            shortened.append(key)
    return pair, shortened


def show_probabilities(title, label, probs):
    st.metric(title, label)
    for name, probability in probs.items():
        st.progress(float(probability), text=f"{name}: {probability:.1%}")


def single_mode(predictor):
    st.header("Analisis satu teks")
    with st.form("single_form"):
        text = st.text_area("Teks / tweet (wajib)", height=160, key="text_input")
        parent = st.text_area("Teks unggahan induk (opsional)", height=100, key="parent_input")
        submitted = st.form_submit_button("Prediksi", type="primary")
    st.caption(f"Batas demonstrasi: {MAX_TEXT_CHARS:,} karakter setelah pra-pemrosesan per kolom. Ini bukan batas token model.")
    if submitted:
        st.session_state.pop("single_result", None)
        if not text.strip():
            st.warning("Teks wajib diisi. Masukkan opini yang ingin dianalisis.")
        else:
            try:
                with st.spinner("Menjalankan prediksi..."):
                    pair, shortened = prepare_input(text, parent)
                    prediction = predictor.predict(**pair)
                    st.session_state["single_result"] = (pair, shortened, prediction, text, parent)
            except (ValueError, RuntimeError) as exc:
                st.error(f"Prediksi gagal: {exc}")
    saved = st.session_state.get("single_result")
    if not saved:
        st.caption("Belum ada prediksi. Isi teks lalu tekan Prediksi.")
        return
    pair, shortened, prediction, source_text, source_parent = saved
    if (text, parent) != (source_text, source_parent):
        st.info("Hasil di bawah berasal dari pengiriman sebelumnya. Tekan Prediksi untuk memperbarui.")
    st.header("Hasil simulasi" if USE_DUMMY else "Hasil prediksi")
    if shortened:
        st.warning(f"Teks dipotong menjadi {MAX_TEXT_CHARS:,} karakter setelah pra-pemrosesan: {', '.join(shortened)}.")
    with st.expander("Teks setelah pra-pemrosesan"):
        st.caption("text")
        st.code(pair["text"], language=None)
        if pair["parent_text"]:
            st.caption("parent_text")
            st.code(pair["parent_text"], language=None)
    left, right = st.columns(2, gap="large")
    with left:
        show_probabilities("Sentimen", prediction["sentiment_label"], prediction["sentiment_probs"])
    with right:
        show_probabilities("Sarkasme", prediction["sarcasm_label"], prediction["sarcasm_probs"])
    st.caption("Probabilitas dummy hanya untuk menguji tampilan. Bukan keyakinan ilmiah atau ukuran akurasi." if USE_DUMMY else "Probabilitas softmax bukan jaminan kebenaran prediksi.")


SENTIMENT_LABELS = ("positif", "negatif", "netral")
SARCASM_LABELS = ("sarkasme", "non-sarkasme")
OUTPUT_COLUMNS = (
    "sentiment_label", "sentiment_prob_positif", "sentiment_prob_negatif", "sentiment_prob_netral",
    "sarcasm_label", "sarcasm_prob_sarkasme", "sarcasm_prob_non_sarkasme",
    "text_preprocessed", "parent_text_preprocessed", "prediction_status", "prediction_note", "model_status",
)


def read_csv_data(raw):
    if not raw or not raw.strip():
        raise ValueError("CSV kosong. Unggah berkas berisi header dan data.")
    if len(raw) > MAX_CSV_BYTES:
        raise ValueError("CSV melebihi batas 10 MiB. Pecah berkas menjadi beberapa bagian.")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV harus berformat UTF-8. Simpan ulang sebagai CSV UTF-8.") from exc
    if "\x00" in content:
        raise ValueError("CSV mengandung karakter NUL. Periksa format berkas.")
    csv.field_size_limit(MAX_CSV_BYTES)
    try:
        rows = list(csv.reader(io.StringIO(content, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError(f"CSV rusak: {exc}. Periksa tanda kutip dan pemisah koma.") from exc
    if not rows or not rows[0]:
        raise ValueError("CSV kosong atau header tidak tersedia.")
    header = rows[0]
    if len(header) != len(set(header)):
        raise ValueError("Nama kolom duplikat. Gunakan nama kolom yang unik.")
    if "text" not in header:
        raise ValueError("Kolom text wajib ada dengan nama persis text. Gunakan pemisah koma.")
    if any(not name for name in header):
        raise ValueError("Nama kolom tidak boleh kosong.")
    collision = set(header).intersection(OUTPUT_COLUMNS)
    if collision:
        raise ValueError(f"Nama kolom berbenturan dengan keluaran: {', '.join(sorted(collision))}. Ubah namanya agar data asli tidak tertimpa.")
    records = []
    for number, row in enumerate(rows[1:], start=2):
        if not row:
            row = [""] * len(header)
        if len(row) != len(header):
            raise ValueError(f"Jumlah kolom pada baris data {number} tidak cocok dengan header.")
        records.append(row)
    if not records:
        raise ValueError("CSV tidak memiliki baris data. Tambahkan teks di bawah header.")
    return pd.DataFrame(records, columns=header, dtype=str)


def predict_frame(frame, predictor, progress=None):
    results = []
    total = len(frame)
    for number, source in enumerate(frame.to_dict("records"), start=1):
        record = dict(source)
        record.update({column: None for column in OUTPUT_COLUMNS})
        record["model_status"] = "dummy_simulasi" if USE_DUMMY else "checkpoint"
        pair, shortened = prepare_input(source["text"], source.get("parent_text", ""))
        record["text_preprocessed"] = pair["text"]
        record["parent_text_preprocessed"] = pair["parent_text"]
        record["prediction_note"] = f"dipotong setelah pra-pemrosesan: {', '.join(shortened)}" if shortened else ""
        if not pair["text"].strip():
            record["prediction_status"] = "dilewati"
            record["prediction_note"] = "text kosong; tidak diprediksi"
        else:
            try:
                prediction = predictor.predict(**pair)
                record["sentiment_label"] = prediction["sentiment_label"]
                record["sarcasm_label"] = prediction["sarcasm_label"]
                record.update({f"sentiment_prob_{label}": probability for label, probability in prediction["sentiment_probs"].items()})
                record.update({f"sarcasm_prob_{label.replace('-', '_')}": probability for label, probability in prediction["sarcasm_probs"].items()})
                record["prediction_status"] = "berhasil"
            except (ValueError, RuntimeError) as exc:
                record["prediction_status"] = "gagal"
                record["prediction_note"] = str(exc)
        results.append(record)
        if progress is not None and (number % max(1, total // 100) == 0 or number == total):
            progress(number, total)
    return pd.DataFrame(results, columns=[*frame.columns, *OUTPUT_COLUMNS])


def distribution(frame, column, labels):
    valid = frame.loc[frame["prediction_status"].eq("berhasil"), column]
    counts = valid.value_counts().reindex(labels, fill_value=0)
    denominator = len(valid)
    return pd.DataFrame({"Label": list(labels), "Jumlah": counts.to_numpy(), "Persen": counts.to_numpy() / denominator * 100 if denominator else [0.0] * len(labels)})


def export_csv(frame):
    return frame.to_csv(index=False, quoting=csv.QUOTE_ALL, lineterminator="\n").encode("utf-8-sig")


def csv_mode(predictor):
    st.header("Analisis berkas CSV")
    st.caption("CSV UTF-8, pemisah koma. Kolom text wajib; parent_text opsional. Maksimum 10 MiB.")
    uploaded = st.file_uploader("Unggah berkas CSV", type=["csv"], key="batch_upload")
    if uploaded is None:
        st.session_state.pop("batch_result", None)
        st.info("Belum ada berkas. Unggah CSV untuk memeriksa kolom dan memprediksi seluruh baris.")
        return
    raw = uploaded.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        frame = read_csv_data(raw)
    except ValueError as exc:
        st.session_state.pop("batch_result", None)
        st.error(str(exc))
        return
    st.caption(f"{len(frame):,} baris terbaca. Teks asli dan urutan baris dipertahankan. Baris text kosong ditandai, bukan diberi prediksi.")
    if st.button("Prediksi CSV", type="primary"):
        status = st.progress(0.0, text="Menyiapkan prediksi baris...")
        with st.spinner("Memproses semua baris CSV..."):
            result = predict_frame(frame, predictor, lambda current, total: status.progress(current / total, text=f"Memproses baris {current:,} dari {total:,}"))
        status.empty()
        st.session_state["batch_result"] = (digest, result)
    saved = st.session_state.get("batch_result")
    if not saved or saved[0] != digest:
        st.info("Berkas valid. Tekan Prediksi CSV untuk menjalankan analisis.")
        return
    result = saved[1]
    valid_count = int(result["prediction_status"].eq("berhasil").sum())
    skipped = int(result["prediction_status"].eq("dilewati").sum())
    failed = int(result["prediction_status"].eq("gagal").sum())
    st.header("Hasil simulasi batch" if USE_DUMMY else "Hasil prediksi batch")
    st.caption(f"{valid_count:,} berhasil, {skipped:,} dilewati, {failed:,} gagal. Total {len(result):,} baris tetap disertakan.")
    if skipped or failed:
        st.warning("Sebagian baris tidak diprediksi. Periksa prediction_status dan prediction_note; probabilitas baris tersebut dibiarkan kosong.")
    truncated = int(result["prediction_note"].fillna("").str.contains("dipotong", regex=False).sum())
    if truncated:
        st.warning(f"{truncated:,} baris memiliki input yang dipotong. Kolom asli tetap utuh; lihat prediction_note dan kolom *_preprocessed.")
    st.dataframe(result, hide_index=True, width="stretch")
    st.subheader("Distribusi label simulasi" if USE_DUMMY else "Distribusi label")
    st.caption("Persentase dihitung dari baris berhasil saja, bukan dari seluruh opini masyarakat.")
    if valid_count:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Sentimen**")
            st.dataframe(distribution(result, "sentiment_label", SENTIMENT_LABELS), hide_index=True, width="stretch", column_config={"Persen": st.column_config.NumberColumn(format="%.1f%%")})
        with right:
            st.markdown("**Sarkasme**")
            st.dataframe(distribution(result, "sarcasm_label", SARCASM_LABELS), hide_index=True, width="stretch", column_config={"Persen": st.column_config.NumberColumn(format="%.1f%%")})
    else:
        st.info("Tidak ada baris berhasil. Distribusi tidak ditampilkan.")
    st.warning("CSV mempertahankan teks asli. Saat membuka data tak tepercaya di Excel, impor kolom teks sebagai Text; jangan jalankan isi yang diawali =, +, -, atau @ sebagai rumus.")
    st.download_button("Unduh CSV hasil", data=export_csv(result), file_name="hasil_simulasi_mbg.csv" if USE_DUMMY else "hasil_prediksi_mbg.csv", mime="text/csv", on_click="ignore")


def main():
    st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="auto")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.header("Mode analisis")
        mode = st.radio("Pilih mode", ["Teks tunggal", "Unggah CSV"])
        st.divider()
        st.markdown("**Status model: dummy (simulasi)**" if USE_DUMMY else "**Status model: checkpoint**")
        st.caption("Checkpoint IndoBERTweet belum digunakan." if USE_DUMMY else f"Lokasi model: {MODEL_DIR}")
        st.info(DISCLAIMER if not USE_DUMMY else DUMMY_DISCLAIMER)
    st.title(TITLE)
    st.caption("Higen Putra Perangin Angin | NIM 211401108 | S1 Ilmu Komputer, Universitas Sumatera Utara")
    st.info(DISCLAIMER if not USE_DUMMY else DUMMY_DISCLAIMER)
    try:
        predictor = get_model()
    except (RuntimeError, OSError, ValueError) as exc:
        st.error(f"Model belum dapat dimuat: {exc}")
        st.stop()
    if USE_DUMMY:
        st.warning("Mode simulasi: label dan probabilitas berasal dari dummy deterministik, bukan model terlatih. Jangan digunakan sebagai hasil penelitian.")
    if mode == "Teks tunggal":
        single_mode(predictor)
    else:
        csv_mode(predictor)


if __name__ == "__main__":
    main()
