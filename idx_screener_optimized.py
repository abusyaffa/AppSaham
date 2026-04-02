# file: idx_screener_optimized.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta          # pip install pandas-ta
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# -------------------------------------------------
# 1. Konstan & Helper
# -------------------------------------------------
MAX_WORKERS = 8                     # sesuaikan dengan jaringan / CPU
CACHE_TTL   = 3600                  # 1 jam (detik)

# Daftar lengkap saham BEI – di‑ambar dari sumber terpercaya.
# Untuk demo kita pakai LQ45 + beberapa tambahan; ganti dengan daftar lengkap bila perlu.
def get_all_beI_tickers():
    """Contoh: menggabungkan LQ45 + beberapa saham lain.
       Dalam produksi, ganti dengan daftar lengkap dari IDX (CSV/API)."""
    lq45 = [
        "ADRO.JK","AKRA.JK","AMMN.JK","ANTM.JK","ARTO.JK","ASTA.JK","BBCA.JK","BBNI.JK","BBRI.JK","BBTN.JK",
        "BMRI.JK","BSDE.JK","CPIN.JK","DFMN.JK","ESSA.JK","EXCL.JK","GOTO.JK","ICBP.JK","INDF.JK","INKP.JK",
        "INTP.JK","ITMG.JK","KLBF.JK","MDKA.JK","MEDC.JK","MIKA.JK","MNCN.JK","PGAS.JK","PNBN.JK","PPRO.JK",
        "PTBA.JK","PWON.JK","RALS.JK","SCMA.JK","SILO.JK","SMGR.JK","TLKM.JK","TPIA.JK","UNTR.JK","UNVR.JK",
        "VOLT.JK","WIKA.JK","WTON.JK"
    ]
    # Tambahkan contoh saham non‑LQ45 agar mencapai ratusan
    extra = [
        "ASII.JK","CPIN.JK","INDY.JK","INKT.JK","MEDc.JK","SCBD.JK","TCPI.JK","UNVR.JK"
    ]  # hanya contoh; sebenarnya Anda dapat memuat dari file CSV.
    return list(set(lq45 + extra))   # hapus duplikat

# -------------------------------------------------
# 2. Fungsi yang di‑cache
# -------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def download_price_data(tickers):
    """Unduh harga historis untuk semua ticker sekaligus."""
    data = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        group_by='ticker',
        auto_adjust=True,
        progress=False,
        threads=True   # yfinance sudah menggunakan threading internally
    )
    # Jika hanya satu ticker, yf.download mengembalikan DataFrame dengan kolom flat.
    # Untuk banyak ticker kita mendapat MultiIndex (ticker, field).
    if isinstance(data.columns, pd.MultiIndex):
        # pisahkan per ticker menjadi dict {ticker: DataFrame}
        out = {}
        for tk in tickers:
            if tk in data['Open'].columns:   # cek apakah kolom ada
                df = data[tk].copy()
                df.dropna(subset=['Open','High','Low','Close','Volume'], inplace=True)
                out[tk] = df
        return out
    else:
        # hanya satu ticker (tidak terjadi dalam penggunaan ini)
        return {tickers[0]: data}

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_fundamentals(ticker):
    """Ambil fundamental sekali per ticker (digunakan dalam thread‑pool)."""
    info = yf.Ticker(ticker).info
    return {
        'ticker': ticker,
        'PE'       : info.get('trailingPE'),
        'PBV'      : info.get('priceToBook'),
        'ROE'      : info.get('returnOnEquity'),
        'DER'      : info.get('debtToEquity'),
        'RevGrowth': info.get('revenueGrowth'),
        'ProfitGrowth': info.get('earningsGrowth')
    }

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_sentiment(ticker):
    """Hitung skor sentimen sederhana dari headline berita."""
    news = yf.Ticker(ticker).news
    if not news:
        return 50.0
    pos = {'naik','untung','growth','profit','dividen','rekor','beli','kuat'}
    neg = {'turun','rugi','loss','pakai','cut','jual','lemah'}
    p=n=0
    for n in news[:10]:
        headline = n.get('title','').lower()
        if any(w in headline for w in pos): p+=1
        if any(w in headline for w in neg): n+=1
    total = p+n
    return 50.0 if total==0 else 50 + 50 * (p-n)/total

# -------------------------------------------------
# 3. Teknikal (vektor)
# -------------------------------------------------
def add_technical_indicators(df):
    """Tambahkan kolom teknikal pada DataFrame harga."""
    df = df.copy()
    df['MA20']  = ta.sma(df['Close'], length=20)
    df['MA50']  = ta.sma(df['Close'], length=50)
    df['MA200'] = ta.sma(df['Close'], length=200)
    df['RSI']   = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df = df.join(macd)
    df['High20'] = df['High'].rolling(20).max().shift(1)
    df['Breakout'] = (df['Close'] > df['High20']).astype(int)
    return df

# -------------------------------------------------
# 4. Skoring
# -------------------------------------------------
def compute_scores(price_dict, fundamentals_list, sentiment_dict):
    """Menghitung skor teknikal, fundamental, dan total."""
    # --- Teknikal ---
    tech_scores = {}
    for tk, df in price_dict.items():
        if df.empty or len(df) < 20:
            continue
        last = df.iloc[-1]
        ma20_sc = 100 if last['Close'] > last['MA20'] else 0
        ma50_sc = 100 if last['Close'] > last['MA50'] else 0
        ma200_sc= 100 if last['Close'] > last['MA200'] else 0
        rsi = last['RSI']
        rsi_sc = 100 if 40 <= rsi <= 60 else (50 - abs(rsi-50))
        macd_sc = 100 if last.get('MACD',0) > last.get('MACDs',0) else 0
        bo_sc   = last['Breakout'] * 100
        tech_scores[tk] = np.nanmean([ma20_sc, ma50_sc, ma200_sc, rsi_sc, macd_sc, bo_sc])

    # --- Fundamental ---
    fund_df = pd.DataFrame(fundamentals_list).set_index('ticker')
    # Normalisasi (min‑max 0‑100); inver untuk metrik yang "semakin rendah semakin baik"
    def norm(series, higher_better=True):
        if series.dropna().empty:
            return pd.Series(np.nan, index=series.index)
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(50, index=series.index)
        scaled = (series - mn) / (mx - mn) * 100
        return scaled if higher_better else 100 - scaled

    fund_norm = pd.DataFrame({
        'PE'       : norm(fund_df['PE'],       higher_better=False),
        'PBV'      : norm(fund_df['PBV'],      higher_better=False),
        'ROE'      : norm(fund_df['ROE'],      higher_better=True),
        'DER'      : norm(fund_df['DER'],      higher_better=False),
        'RevGrowth': norm(fund_df['RevGrowth'],higher_better=True),
        'ProfitGrowth': norm(fund_df['ProfitGrowth'],higher_better=True)
    })
    fund_scores = fund_norm.mean(axis=1)   # rata‑rata semua dimensi fundamental

    # --- Sentimen ---
    sent_scores = pd.Series(sentiment_dict)

    # Gabungkan hanya ticker yang ada di semua tiga kamus
    common = set(tech_scores) & set(fund_scores.index) & set(sent_scores.index)
    tech = pd.Series(tech_scores).loc[common]
    fund = fund_scores.loc[common]
    sent = sent_scores.loc[common]

    total = 0.4*tech + 0.4*fund + 0.2*sent
    result = pd.DataFrame({
        'Technical'   : tech,
        'Fundamental' : fund,
        'Sentiment'   : sent,
        'TotalScore'  : total
    }).sort_values('TotalScore', ascending=False)
    return result

# -------------------------------------------------
# 5. Streamlit UI (sama seperti sebelumnya, tapi lebih ringan)
# -------------------------------------------------
st.set_page_config(page_title="IDX Screener – Optimized", layout="wide")
st.title("🇮🇩 IDX Stock Screener – Optimized for Hundreds of Tickers")
st.caption("Sumber data: Yahoo Finance (ticker .JK). Teknik optimasi: batch download, thread‑pool, caching.")

if st.button("🔄 Jalankan Screening (Optimized)"):
    with st.spinner("Mengunduh data dan menghitung skor…"):
        tickers = get_all_beI_tickers()

        # 1️⃣ Batch harga
        price_dict = download_price_data(tickers)

        # 2️⃣ Fundamental & Sentimen via thread‑pool
        fundamentals = []
        sentiments   = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            #_submit semua ticker
            future_to_ticker = {
                executor.submit(fetch_fundamentals, tk): tk for tk in tickers
            }
            for fut in as_completed(future_to_ticker):
                tk = future_to_ticker[fut]
                try:
                    fundamentals.append(fut.result())
                except Exception as e:
                    st.warning(f"Gagal ambil fundamental {tk}: {e}")

            future_to_ticker_sent = {
                executor.submit(fetch_sentiment, tk): tk for tk in tickers
            }
            for fut in as_completed(future_to_ticker_sent):
                tk = future_to_ticker_sent[fut]
                try:
                    sentiments[tk] = fut.result()
                except Exception as e:
                    sentiments[tk] = 50.0   # fallback netral

        # 3️⃣ Skoring
        scores = compute_scores(price_dict, fundamentals, sentiments)

    # ------------------- Tampilkan hasil -------------------
    st.subheader("🏆 Top 10 Saham Berdasarkan Total Score")
    top10 = scores.head(10).reset_index()
    top10.columns = ['Ticker','Technical','Fundamental','Sentiment','TotalScore']
    st.dataframe(
        top10.style.format({
            'Technical'  : "{:.1f}",
            'Fundamental': "{:.1f}",
            'Sentiment'  : "{:.1f}",
            'TotalScore' : "{:.1f}"
        }),
        use_container_width=True
    )

    # Pilih saham untuk detail chart
    chosen = st.selectbox("Lihat chart saham:", options=top10['Ticker'])
    if chosen:
        df_chart = price_dict[chosen].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_chart['Date'],
            open=df_chart['Open'], high=df_chart['High'],
            low=df_chart['Low'],  close=df_chart['Close'],
            name='Harga'
        ))
        fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA20'],
                                 line=dict(color='orange', width=1), name='MA20'))
        fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA50'],
                                 line=dict(color='blue', width=1), name='MA50'))
        fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA200'],
                                 line=dict(color='purple', width=1), name='MA200'))
        fig.update_layout(
            title=f"{chosen} – Harga + MA",
            xaxis_title="Tanggal",
            yaxis_title="Harga (IDR)",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Alasan singkat
        st.markdown(f"**Alasan skor tinggi untuk {chosen}:**")
        reasons = []
        row = scores.loc[chosen]
        if row['Technical'] >= 70: reasons.append("- Teknikal kuat (harga di atas MA, RSI seimbang, MACD bullish, breakout).")
        if row['Fundamental'] >= 70: reasons.append("- Fundamental menarik (PER/PBV relatif rendah, ROE tinggi, utang sehat, pertumbuhan pendapatan/laba positif).")
        if row['Sentiment'] >= 60: reasons.append("- Sentimen berita cenderung positif.")
        if not reasons: reasons.append("- Skor seimbang di seluruh dimensi.")
        st.markdown("\n".join(reasons))

st.markdown("---")
st.caption("Catatan: Untuk produksi dengan ratusan‑ribu ticker, pertimbangkan menyimpan hasil download harga ke file lokal (parquet/feather) dan hanya memperbarui incremento harian.")