"""
Studi Kasus Sesi 6: Mengklasifikasikan Penyebab Langsung Peringatan Deforestasi dari Citra Satelit dengan AI
(simulasi citra Sentinel-1 radar + Sentinel-2 optik, dengan dan tanpa awan)
Berdasarkan webinar "Attributing direct drivers to deforestation alerts with satellite imagery and AI"
(Bart Slagter, Wageningen University & Research).

Ide dari materi yang diterapkan:
  - Peringatan deforestasi hanya bilang "hutan berubah"; AI menambahkan PENYEBAB LANGSUNG-nya
    (tujuh kelas: pertanian skala kecil/besar, jalan, penebangan selektif, pertambangan, kebakaran, gangguan alami)
  - Input = tumpukan pita Sentinel-1 (2 pita radar) + Sentinel-2 (4 pita optik) sebagai citra multi-pita
  - Radar menembus awan, optik kaya informasi tetapi terhalang awan -> keduanya saling melengkapi
  - Benchmark metode konvensional dulu, baru bandingkan dengan deep learning (CNN)
  - Akurasi total menyesatkan karena kelas pertanian dominan; macro-F1 menimbang tiap kelas sama
  - Kebakaran vs banjir/gangguan alami paling sering tertukar; menambah contoh latih terarah membantu
  - Label "tidak yakin" untuk prediksi berkeyakinan rendah
  - Keterbatasan: hanya penyebab langsung; pembedaan lebih rinci butuh data lapangan

DATA SINTETIS: citra dibuat oleh simulator sederhana (bukan citra satelit asli), jadi angka hasilnya
hanya memperagakan konsep dan TIDAK boleh dianggap akurasi pada citra nyata.
CNN yang dipakai adalah CNN mini (1 lapisan konvolusi) ditulis dengan NumPy agar tanpa instalasi tambahan;
untuk penelitian sungguhan gunakan PyTorch atau TensorFlow.
Jalankan:  python case_study_forest.py          Opsi: --epochs 30 --seed 42
"""
import argparse, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

OUT, SEED, H, W = "output", 42, 24, 24
CLASSES = ["Pertanian skala kecil", "Pertanian skala besar", "Jalan", "Penebangan selektif", "Pertambangan", "Kebakaran hutan", "Gangguan alami"]
SHORT = ["Tani kecil", "Tani besar", "Jalan", "Tebang selektif", "Tambang", "Kebakaran", "Alami"]
PROPORSI = np.array([.28, .22, .10, .12, .04, .12, .12])         # pertanian dominan, seperti pada materi
FOREST = np.array([-7.5, -13.5, .03, .05, .03, .35])             # VV(dB), VH(dB), B2, B3, B4, B8 (hutan sehat)
SIG = {"bare": [-9.5, -17, .06, .09, .11, .22], "burn": [-8.5, -14.5, .03, .04, .06, .12], "mine": [-6, -14, .14, .17, .21, .24],
       "pond": [-19, -25, .05, .06, .04, .03], "flood": [-16, -21, .04, .05, .03, .08], "treefall": [-8.5, -15, .04, .06, .05, .27]}
YY, XX = np.mgrid[:H, :W]
SUMMARY = {}


def savefig(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}", dpi=130); plt.close()


# ------------------------------------------------------------- 1. SIMULATOR CITRA SATELIT
def blob(rng, r, irr=0.35):
    cx, cy = rng.uniform(4, W - 4), rng.uniform(4, H - 4); ang = np.arctan2(YY - cy, XX - cx)
    mod = 1 + irr * (rng.normal() * np.sin(ang + rng.uniform(0, 6.28)) + 0.6 * rng.normal() * np.sin(2 * ang + rng.uniform(0, 6.28)))
    return np.hypot(YY - cy, XX - cx) < r * np.clip(mod, 0.4, 1.8)


def line(rng, width):
    p = rng.uniform(0, 2 * np.pi); cx, cy = rng.uniform(6, W - 6, 2)
    d = np.cos(p) * (YY - cy) - np.sin(p) * (XX - cx)
    wob = rng.uniform(0, 1) * 0.8 * np.sin((XX * np.cos(p) + YY * np.sin(p)) / 3 + rng.uniform(0, 6.28))
    return np.abs(d + wob) < width / 2


def gaps(rng, n, rmin, rmax, near=None):
    m = np.zeros((H, W), bool)
    for _ in range(n):
        if near is not None and near.any():
            ys, xs = np.nonzero(near); k = rng.integers(len(ys)); cy, cx = ys[k] + rng.normal(0, 2), xs[k] + rng.normal(0, 2)
        else:
            cy, cx = rng.uniform(1, H - 1), rng.uniform(1, W - 1)
        m |= np.hypot(YY - cy, XX - cx) < rng.uniform(rmin, rmax)
    return m


def river(rng, width):
    ctr = rng.uniform(6, W - 6) + rng.uniform(2, 5) * np.sin(XX * rng.uniform(0.15, 0.3) + rng.uniform(0, 6.28))
    m = np.abs(YY - ctr) < width / 2
    return m.T if rng.random() < 0.5 else m


def make_patch(rng, cls, cloud_p):
    img = np.tile(FOREST[:, None, None], (1, H, W)).astype(float)
    img += rng.normal(0, [0.4, 0.4, .003, .004, .003, .02])[:, None, None]

    def paint(mask, sig, strength=1.0):
        s = np.array(SIG[sig]) + rng.normal(0, [1.3, 1.3, .012, .015, .02, .04]); s = FOREST + strength * (s - FOREST)
        for c in range(6): img[c][mask] = s[c]

    if cls == 0:
        m = np.zeros((H, W), bool)
        for _ in range(rng.integers(1, 4)): m |= blob(rng, rng.uniform(2, 4))
        paint(m, "bare")
    elif cls == 1: paint(blob(rng, rng.uniform(7, 11), 0.25), "bare")
    elif cls == 2: paint(line(rng, rng.uniform(1.0, 2.2)), "bare")
    elif cls == 3:
        ln = line(rng, 1.0); paint(ln | gaps(rng, rng.integers(4, 9), 0.8, 1.6, near=ln), "bare")
    elif cls == 4:
        m = blob(rng, rng.uniform(4, 7)); paint(m, "mine"); paint(gaps(rng, rng.integers(1, 4), 1.0, 2.0, near=m) & m, "pond")
    elif cls == 5:
        if rng.random() < 0.3: paint(river(rng, rng.uniform(2, 4)), "flood", rng.uniform(0.3, 1.0))      # kebakaran di dataran banjir
        paint(blob(rng, rng.uniform(5, 10), 0.6), "burn")
    else:
        if rng.random() < 0.5: paint(river(rng, rng.uniform(3, 6)), "flood", rng.uniform(0.3, 1.0))      # banjir (kanopi meredam sinyal radar)
        else: paint(gaps(rng, rng.integers(5, 12), 1.0, 2.2) | blob(rng, 3), "treefall")                  # pohon tumbang akibat badai
    img[:2] += rng.normal(0, 1.4, (2, H, W))                                                               # speckle radar
    img[2:] = np.clip(img[2:] * (1 + rng.normal(0, 0.06, (4, H, W))) + rng.normal(0, 0.008, (4, H, W)), 0.005, 1.0)
    if rng.random() < cloud_p:                                                                              # awan menutupi pita optik
        field = ndimage.gaussian_filter(rng.normal(size=(H, W)), 3); thr = np.quantile(field, 1 - rng.uniform(0.3, 0.95))
        alpha = ndimage.gaussian_filter((field > thr).astype(float), 1) * rng.uniform(0.85, 1.0)
        cloud = rng.uniform(0.45, 0.75, (4, 1, 1)) * (1 + rng.normal(0, 0.05, (4, H, W)))
        img[2:] = img[2:] * (1 - alpha) + cloud * alpha
    return np.moveaxis(img, 0, -1).astype(np.float32)


def generate(rng, n, cloud_p, extra_classes=None):
    if extra_classes is None:
        counts = np.maximum(np.round(n * PROPORSI).astype(int), 15)
    else:
        counts = np.zeros(7, int); counts[list(extra_classes)] = n
    y = np.repeat(np.arange(7), counts); rng.shuffle(y)
    return np.stack([make_patch(rng, c, cloud_p) for c in y]), y


# ------------------------------------------------------------- 2. FITUR BUATAN TANGAN (METODE KONVENSIONAL)
def mask_feats(m):
    lab, n = ndimage.label(m)
    if n == 0: return [0.0, 0, 0.0, 1.0]
    sizes = np.bincount(lab.ravel())[1:]; big = sizes.argmax() + 1
    ys, xs = np.nonzero(lab == big)
    el = 1.0
    if len(ys) >= 3:
        ev = np.linalg.eigvalsh(np.cov(np.vstack([ys, xs]))); el = min(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)), 10.0)
    return [m.mean(), n, sizes.max() / m.size, el]


def features(X, sensors="s1s2"):
    rows = []
    for p in X:
        f = []
        if "s1" in sensors:
            vv, vh = ndimage.uniform_filter(p[..., 0], 3), ndimage.uniform_filter(p[..., 1], 3)         # peredaman speckle
            f += [vv.mean(), vv.std(), vh.mean(), vh.std(), (vh - vv).mean(), vv.min(), vv.max(), vh.min()]
            f += mask_feats(vh < -15.5) + mask_feats(vv < -11.5)
        if "s2" in sensors:
            b2, b3, b4, b8 = p[..., 2], p[..., 3], p[..., 4], p[..., 5]; ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
            f += [b2.mean(), b3.mean(), b4.mean(), b8.mean(), b8.std(), ndvi.mean(), ndvi.min(), ndvi.std(), (b2 > 0.25).mean()]
            f += mask_feats(ndvi < 0.55) + mask_feats(b8 < 0.15)
        rows.append(f)
    return np.array(rows, float)


# ------------------------------------------------------------- 3. CNN MINI (NumPy)
class MiniCNN:
    """Konvolusi 3x3 -> ReLU -> pooling global (rata-rata + maksimum) -> lapisan padat -> softmax. Dilatih dengan Adam."""
    def __init__(self, c_in, n_filters=12, n_classes=7, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed); F = n_filters; self.F, self.dt = F, dtype
        self.p = {"W": (rng.normal(0, np.sqrt(2 / (9 * c_in)), (9 * c_in, F))).astype(dtype), "b": np.zeros(F, dtype),
                  "V": (rng.normal(0, np.sqrt(1 / (2 * F)), (2 * F, n_classes))).astype(dtype), "c": np.zeros(n_classes, dtype)}

    def forward(self, X):
        N = len(X); win = sliding_window_view(X, (3, 3), axis=(1, 2)); h, w = win.shape[1:3]
        cols = win.transpose(0, 1, 2, 4, 5, 3).reshape(N * h * w, -1).astype(self.dt)
        Z = (cols @ self.p["W"] + self.p["b"]).reshape(N, h * w, self.F); A = np.maximum(Z, 0)
        idx = A.argmax(1); pooled = np.concatenate([A.mean(1), A.max(1)], 1)
        logits = pooled @ self.p["V"] + self.p["c"]
        self.cache = (cols, Z, idx, pooled, N, h * w)
        e = np.exp(logits - logits.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

    def backward(self, prob, y, sw):
        cols, Z, idx, pooled, N, hw = self.cache; F = self.F
        d = prob.copy(); d[np.arange(N), y] -= 1; d *= (sw / sw.sum())[:, None]
        g = {"V": pooled.T @ d, "c": d.sum(0)}; dp = d @ self.p["V"].T
        dA = np.zeros((N, hw, F), self.dt); dA += dp[:, None, :F] / hw
        dA[np.arange(N)[:, None], idx, np.arange(F)[None, :]] += dp[:, F:]
        dZ = (dA * (Z > 0)).reshape(-1, F); g["W"] = cols.T @ dZ; g["b"] = dZ.sum(0)
        return g

    def fit(self, X, y, epochs=30, batch=64, lr=3e-3, class_weight=None, seed=0, log=None):
        rng = np.random.default_rng(seed); m = {k: np.zeros_like(v) for k, v in self.p.items()}; v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        t, losses = 0, []
        for ep in range(epochs):
            order = rng.permutation(len(X)); tot = 0.0
            for s in range(0, len(X), batch):
                i = order[s:s + batch]; sw = class_weight[y[i]] if class_weight is not None else np.ones(len(i))
                prob = self.forward(X[i]); tot += -(sw * np.log(prob[np.arange(len(i)), y[i]] + 1e-9)).sum() / sw.sum() * len(i)
                g = self.backward(prob, y[i], sw); t += 1
                for k in self.p:
                    m[k] = 0.9 * m[k] + 0.1 * g[k]; v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                    self.p[k] -= (lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)).astype(self.dt)
            losses.append(tot / len(X))
            if log and (ep + 1) % 5 == 0: print(f"   epoch {ep+1:>2}/{epochs}  loss = {losses[-1]:.4f}")
        return losses

    def predict_proba(self, X, batch=256):
        return np.vstack([self.forward(X[s:s + batch]) for s in range(0, len(X), batch)])


def gradient_check():
    """Uji turunan: gradien analitik harus sama dengan gradien numerik (pastikan CNN buatan sendiri benar)."""
    rng = np.random.default_rng(0); net = MiniCNN(6, 3, 7, seed=1, dtype=np.float64)
    X = rng.normal(size=(5, 10, 10, 6)); y = rng.integers(0, 7, 5); sw = np.ones(5)
    loss = lambda: -np.log(net.forward(X)[np.arange(5), y] + 1e-12).mean()
    net.forward(X); g = net.backward(net.forward(X), y, sw); worst = 0.0
    for k in net.p:
        flat = net.p[k].reshape(-1)
        for j in rng.choice(len(flat), min(4, len(flat)), replace=False):
            old = flat[j]; flat[j] = old + 1e-6; l1 = loss(); flat[j] = old - 1e-6; l2 = loss(); flat[j] = old
            num = (l1 - l2) / 2e-6; ana = g[k].reshape(-1)[j]; worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


# ------------------------------------------------------------- 4. EVALUASI
def metrics(y, pred):
    return accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0), f1_score(y, pred, average=None, labels=range(7), zero_division=0)


def main(epochs, seed):
    global SEED; SEED = seed; rng = np.random.default_rng(seed)
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 170, "display.max_columns", 20)

    print("== 1) SIMULASI CITRA SATELIT =="); t0 = time.time()
    Xtr, ytr = generate(rng, 3000, 0.4)                    # latih: 40% citra berawan
    Xmix, ymix = generate(rng, 1200, 0.4)                  # uji utama
    Xclr, yclr = generate(rng, 1000, 0.0)                  # uji tanpa awan
    Xcld, ycld = generate(rng, 1000, 1.0)                  # uji semua berawan
    print(f"Patch {H}x{W} piksel, 6 pita (2 radar Sentinel-1 + 4 optik Sentinel-2). Latih {len(Xtr)}, uji campuran {len(Xmix)}, "
          f"uji tanpa awan {len(Xclr)}, uji berawan {len(Xcld)} ({time.time()-t0:.0f} detik)")
    print("Jumlah per kelas (latih):", dict(zip(SHORT, np.bincount(ytr))))
    fig, ax = plt.subplots(7, 4, figsize=(8, 14)); r2 = np.random.default_rng(7)
    rgb = lambda p: np.clip(p[..., [4, 3, 2]] * 4, 0, 1)
    for c in range(7):
        a, b = make_patch(r2, c, 0.0), make_patch(r2, c, 1.0)
        for j, (im, ttl) in enumerate([(rgb(a), "Optik (RGB)"), (a[..., 0], "Radar VV"), (a[..., 1], "Radar VH"), (rgb(b), "Optik berawan")]):
            ax[c, j].imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=None if im.ndim == 3 else -22, vmax=None if im.ndim == 3 else -4); ax[c, j].axis("off")
            if c == 0: ax[c, j].set_title(ttl, fontsize=8)
        ax[c, 0].text(-2, 12, SHORT[c], fontsize=8, ha="right", va="center")
    savefig("01_contoh_citra_simulasi.png")

    print("\n== 2) METODE KONVENSIONAL: RANDOM FOREST PADA FITUR BUATAN TANGAN ==")
    Ftr, Fmix, Fclr, Fcld = (features(x) for x in (Xtr, Xmix, Xclr, Xcld))
    rf = RandomForestClassifier(300, class_weight="balanced_subsample", random_state=seed, n_jobs=-1).fit(Ftr, ytr)
    base = DummyClassifier(strategy="most_frequent").fit(Ftr, ytr)
    prf = rf.predict_proba(Fmix)

    print("\n== 3) CNN MINI (NumPy) ==")
    print(f"Uji turunan (gradient check): galat relatif terbesar = {gradient_check():.1e} (harus sangat kecil)")
    mu, sd = Xtr.mean((0, 1, 2)), Xtr.std((0, 1, 2)) + 1e-6; norm = lambda x: ((x - mu) / sd).astype(np.float32)
    cw = len(ytr) / (7 * np.bincount(ytr)); cw = (cw / cw.mean()).astype(np.float32)
    cnn = MiniCNN(6, 12, 7, seed=seed); t0 = time.time()
    losses = cnn.fit(norm(Xtr), ytr, epochs=epochs, class_weight=cw, seed=seed, log=True)
    print(f"Pelatihan selesai dalam {time.time()-t0:.0f} detik")
    pcnn = cnn.predict_proba(norm(Xmix))
    plt.figure(figsize=(5, 3.5)); plt.plot(range(1, epochs + 1), losses); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Kurva pelatihan CNN mini")
    savefig("02_kurva_pelatihan_cnn.png")

    print("\n== 4) EVALUASI: AKURASI VS MACRO-F1 ==")
    rows, perclass = [], {}
    for name, pred in [("Baseline (tebak mayoritas)", base.predict(Fmix)), ("Random Forest (fitur buatan)", prf.argmax(1)), ("CNN mini (citra 6 pita)", pcnn.argmax(1))]:
        acc, mf1, pc = metrics(ymix, pred); rows.append({"model": name, "accuracy": acc, "macro_F1": mf1}); perclass[name] = pc
    res = pd.DataFrame(rows).set_index("model").round(3)
    pcdf = pd.DataFrame(perclass, index=SHORT).round(3)
    res.to_csv(f"{OUT}/metrik_utama.csv"); pcdf.to_csv(f"{OUT}/f1_per_kelas.csv"); print(res.to_string()); print("\nF1 per kelas:\n", pcdf.to_string())
    pcdf.drop(columns="Baseline (tebak mayoritas)").plot(kind="bar", figsize=(9, 4)); plt.ylabel("F1"); plt.xticks(rotation=25, ha="right"); plt.ylim(0, 1.05)
    plt.title("F1 per kelas: kelas langka/ambigu lebih sulit"); savefig("03_f1_per_kelas.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, (nm, pr) in zip(ax, [("Random Forest", prf), ("CNN mini", pcnn)]):
        cm = confusion_matrix(ymix, pr.argmax(1), labels=range(7), normalize="true"); a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7): a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if cm[i, j] > .5 else "black")
        a.set_xticks(range(7), SHORT, rotation=60, fontsize=7); a.set_yticks(range(7), SHORT, fontsize=7); a.set_title(f"{nm} (baris = kelas sebenarnya)", fontsize=9)
    savefig("04_confusion_matrix.png")
    SUMMARY["Akurasi vs macro-F1"] = (f"baseline akurasi {res.iloc[0,0]:.2f} tapi macro-F1 {res.iloc[0,1]:.2f}; RF {res.iloc[1,0]:.2f}/{res.iloc[1,1]:.2f}; CNN {res.iloc[2,0]:.2f}/{res.iloc[2,1]:.2f}")

    print("\n== 5) FUSI SENSOR DAN AWAN (Random Forest) ==")
    ab = []
    for sensors, label in [("s1", "Radar saja (S1)"), ("s2", "Optik saja (S2)"), ("s1s2", "Radar + optik (S1+S2)")]:
        m = RandomForestClassifier(300, class_weight="balanced_subsample", random_state=seed, n_jobs=-1).fit(features(Xtr, sensors), ytr)
        for cond, X, y in [("tanpa awan", Xclr, yclr), ("berawan", Xcld, ycld)]:
            acc, mf1, _ = metrics(y, m.predict(features(X, sensors))); ab.append({"sensor": label, "kondisi uji": cond, "accuracy": acc, "macro_F1": mf1})
    ab = pd.DataFrame(ab).round(3); ab.to_csv(f"{OUT}/ablasi_sensor_awan.csv", index=False)
    pv = ab.pivot(index="sensor", columns="kondisi uji", values="macro_F1")[["tanpa awan", "berawan"]]; print(pv.to_string())
    cnn_c = [metrics(y, cnn.predict_proba(norm(X)).argmax(1))[1] for X, y in [(Xclr, yclr), (Xcld, ycld)]]
    print(f"CNN mini (S1+S2): macro-F1 tanpa awan {cnn_c[0]:.3f}, berawan {cnn_c[1]:.3f}")
    pv.plot(kind="bar", figsize=(7, 4)); plt.ylabel("macro-F1"); plt.xticks(rotation=0); plt.ylim(0, 1.05); plt.title("Radar menutupi kelemahan optik saat berawan")
    savefig("05_ablasi_sensor_awan.png")
    SUMMARY["Fusi sensor dan awan"] = (f"macro-F1 berawan: radar {pv.loc['Radar saja (S1)','berawan']:.2f}, optik {pv.loc['Optik saja (S2)','berawan']:.2f}, "
                                      f"gabungan {pv.loc['Radar + optik (S1+S2)','berawan']:.2f}")

    print("\n== 6) LABEL 'TIDAK YAKIN' ==")
    th = np.arange(0.3, 0.96, 0.05); cur = {}
    for nm, pr in [("Random Forest", prf), ("CNN mini", pcnn)]:
        pm, pc = pr.max(1), pr.argmax(1)
        cur[nm] = [(float((pm >= t).mean()), float((pc[pm >= t] == ymix[pm >= t]).mean()) if (pm >= t).any() else np.nan) for t in th]
    tabel = pd.DataFrame([{"model": nm, "ambang": round(t, 2), "cakupan": c, "akurasi_yang_diterima": a} for nm, v in cur.items() for t, (c, a) in zip(th, v)]).round(3)
    tabel.to_csv(f"{OUT}/tidak_yakin.csv", index=False)
    for nm in cur:
        i = int(np.argmin(np.abs(th - 0.7))); c, a = cur[nm][i]; print(f"{nm}: ambang 0,70 -> {c:.0%} prediksi diterima dengan akurasi {a:.0%}; sisanya diberi label 'tidak yakin'")
    plt.figure(figsize=(5.5, 4))
    for nm, v in cur.items(): plt.plot([x[0] for x in v], [x[1] for x in v], "o-", markersize=3, label=nm)
    plt.xlabel("Cakupan (proporsi prediksi yang diterima)"); plt.ylabel("Akurasi pada prediksi yang diterima"); plt.legend(); plt.title("Menolak prediksi ragu menaikkan akurasi")
    savefig("06_tidak_yakin.png")
    SUMMARY["Label tidak yakin"] = "ambang 0,70: " + "; ".join(f"{nm} {cur[nm][int(np.argmin(np.abs(th-0.7)))][0]:.0%} diterima, akurasi {cur[nm][int(np.argmin(np.abs(th-0.7)))][1]:.0%}" for nm in cur)

    print("\n== 7) MENAMBAH CONTOH LATIH TERARAH (kebakaran dan gangguan alami) ==")
    rows = []
    for rep in range(3):
        r = np.random.default_rng(1000 + rep); Xe, ye = generate(r, 150, 0.4, extra_classes=(5, 6))
        Fx, yx = np.vstack([Ftr, features(Xe)]), np.r_[ytr, ye]
        for nm, F, y in [("latih awal", Ftr, ytr), ("+150 contoh kebakaran dan alami", Fx, yx)]:
            m = RandomForestClassifier(300, class_weight="balanced_subsample", random_state=rep, n_jobs=-1).fit(F, y)
            _, mf1, pc = metrics(ymix, m.predict(Fmix)); rows.append({"skema": nm, "rep": rep, "F1_kebakaran": pc[5], "F1_alami": pc[6], "macro_F1": mf1})
    rt = pd.DataFrame(rows).groupby("skema", sort=False)[["F1_kebakaran", "F1_alami", "macro_F1"]].mean().round(3); rt.to_csv(f"{OUT}/retraining_terarah.csv"); print(rt.to_string())
    SUMMARY["Retraining terarah"] = (f"F1 kebakaran {rt.iloc[0,0]:.2f} -> {rt.iloc[1,0]:.2f}; F1 alami {rt.iloc[0,1]:.2f} -> {rt.iloc[1,1]:.2f}; macro-F1 {rt.iloc[0,2]:.2f} -> {rt.iloc[1,2]:.2f} (rata-rata 3 pengulangan)")

    md = ["# Ringkasan Hasil (dibuat otomatis; DATA SIMULASI, bukan citra satelit asli)\n"] + [f"- **{k}:** {v}" for k, v in SUMMARY.items()]
    open(f"{OUT}/ringkasan.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); main(a.epochs, a.seed)
