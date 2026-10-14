"""
Confronto modello fisico - dati sperimentali (riscaldamento ohmico multifase).

Legge un file Excel con colonne
    Configurazione | Tensione | Concentrazione salina | Ttempo | Temperatura 1 | Temperatura 2 | Temperatura 3
(una riga per istante, piu' prove nello stesso foglio), traduce gli attributi di ciascuna prova nei
parametri del modello (multifase_engine) e confronta le temperature simulate con quelle misurate.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import multifase_engine as me

# ---------------------------------------------------------------------------
# Lettura e validazione del file
# ---------------------------------------------------------------------------
COLONNE = ['configurazione', 'tensione', 'sale', 'tempo', 'T1', 'T2', 'T3']


def _normalizza_nome(c):
    k = str(c).strip().lower().replace(' ', '').replace('_', '')
    if k.startswith('config'):
        return 'configurazione'
    if k.startswith('tension') or k in ('voltage', 'volt', 'v'):
        return 'tensione'
    if k.startswith('concentr') or k.startswith('sal') or 'salt' in k:
        return 'sale'
    if k in ('ttempo', 'tempo', 'time', 'tempos', 't[s]', 'ts'):
        return 'tempo'
    for i in (1, 2, 3):
        if k in (f'temperatura{i}', f'temp{i}', f't{i}', f'temperature{i}', f'sonda{i}', f'p{i}'):
            return f'T{i}'
    return None


def leggi_dati(sorgente):
    """Restituisce (DataFrame normalizzato, lista di avvisi). Solleva ValueError se mancano colonne."""
    df = pd.read_excel(sorgente)
    ren = {}
    for c in df.columns:
        n = _normalizza_nome(c)
        if n and n not in ren.values():
            ren[c] = n
    df = df.rename(columns=ren)
    mancanti = [c for c in COLONNE if c not in df.columns]
    if mancanti:
        raise ValueError(f"Colonne non riconosciute: {mancanti}. Colonne trovate nel file: {list(df.columns)}")
    df = df[COLONNE].copy()
    avvisi = []
    for c in COLONNE[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['configurazione'] = df['configurazione'].astype(str).str.strip()
    n0 = len(df)
    nan_righe = df[df[['tensione', 'sale', 'tempo']].isna().any(axis=1)]
    if len(nan_righe):
        avvisi.append(f"{len(nan_righe)} righe senza tensione/sale/tempo sono state scartate.")
        df = df.drop(nan_righe.index)
    incomplete = df[df[['T1', 'T2', 'T3']].isna().any(axis=1)]
    if len(incomplete):
        elenco = incomplete.groupby(['configurazione', 'tensione', 'sale']).size()
        for (c, v, s), n in elenco.items():
            avvisi.append(f"Prova {c} / {v:g} V / sale {s:g}: {n} riga/righe con temperature mancanti, escluse dal confronto.")
        df = df.drop(incomplete.index)
    df = df.sort_values(['configurazione', 'tensione', 'sale', 'tempo']).reset_index(drop=True)
    return df, avvisi


def elenco_prove(df):
    g = df.groupby(['configurazione', 'tensione', 'sale'])
    r = g.agg(punti=('tempo', 'size'), t_max=('tempo', 'max'), dt=('tempo', lambda x: x.diff().median())).reset_index()
    r['T_ini'] = g[['T1', 'T2', 'T3']].first().mean(axis=1).values
    r['T_fin_media'] = g[['T1', 'T2', 'T3']].last().mean(axis=1).values
    r['id'] = r.apply(lambda x: f"{x['configurazione']} | {x['tensione']:g} V | sale {x['sale']:g}", axis=1)
    return r


def estrai_prova(df, conf, V, sale):
    d = df[(df['configurazione'] == conf) & (df['tensione'] == V) & (np.isclose(df['sale'], sale))]
    return d.sort_values('tempo').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mappatura attributi -> modello
# ---------------------------------------------------------------------------
# posizioni (P1, P2, P3) occupate da una polpetta, dedotte dal nome della configurazione.
# 'Middle' e 'Side' (una sola polpetta, al centro o a un lato) sono dedotte dalle temperature
# iniziali (la sonda dentro la polpetta parte piu' fredda) e VANNO CONFERMATE.
OCCUPAZIONE_DEFAULT = {
    'tris': (True, True, True), '3pirp': (True, True, True),
    'symm': (True, False, True), '2psirp': (True, False, True),
    'asymm': (True, True, False), '2pairp': (True, True, False),
    'middle': (False, True, False),
    'side': (True, False, False),
    'rp': (False, False, False), 'omogeneo': (False, False, False),
}


def occupazione_default(nome):
    return OCCUPAZIONE_DEFAULT.get(str(nome).strip().lower(), (False, False, False))


def composizione_da_sale(sale, burro=5.8, acqua=72.5):
    """Ricetta RPFB (Tab. 4.3, composizioni 7-9): burro e acqua costanti, fiocchi + sale = 100 - burro - acqua."""
    fiocchi = 100.0 - burro - acqua - sale
    if fiocchi <= 0:
        raise ValueError("La somma burro + acqua + sale supera il 100%.")
    return dict(w_burro=burro, w_sale=sale, w_fiocchi=fiocchi, w_acqua=acqua)


def regressione_riferimento(sale, acqua):
    """Regressione sigma(T) di letteratura piu' vicina per rapporto sale/acqua (Tab. 5.1 e 6.1)."""
    r = sale / acqua
    best, dist = None, 1e9
    for i, c in me.COMPOSIZIONI.items():
        if c['sigma'] is None:
            continue
        d = abs(c['sale'] / c['acqua'] - r)
        if d < dist:
            best, dist = i, d
    a, b = me.COMPOSIZIONI[best]['sigma'][:2]
    return best, a, b


def alpha_da_regressione(a, b):
    s20 = 20.0 * a + b
    return a / s20, s20


def sigma_da_s20_alpha(s20, alpha):
    """sigma(T) = s20 * (1 + alpha (T - 20))  ->  (a, b) con sigma = a T + b."""
    a = s20 * alpha
    return a, s20 - 20.0 * a


class Impostazioni:
    """Impostazioni di modello comuni a tutte le prove."""
    def __init__(self, cella='UNISA (10 x 5.1 x 3.7 cm)', Lx_mm=None, tensione_rms=True,
                 burro=5.8, acqua=72.5, h_modo='naturale_irraggiamento', T_amb=None,
                 prop_modo='choi_okos_T'):
        self.cella = cella
        self.Lx_mm = Lx_mm
        self.tensione_rms = tensione_rms
        self.burro = burro
        self.acqua = acqua
        self.h_modo = h_modo
        self.T_amb = T_amb
        self.prop_modo = prop_modo
        self.occupazione = {}       # nome configurazione -> (b1, b2, b3)


def _geometria_prova(imp, conf):
    occ = imp.occupazione.get(conf, occupazione_default(conf))
    ov = dict(occupazione=list(occ))
    if imp.Lx_mm:
        ov['Lx'] = imp.Lx_mm / 1000.0
    return me.geometria(imp.cella, 'RP (solo purea)', ov), occ


def prepara_prova(d, imp, sigma_ab):
    """Costruisce (geo, p) per una prova a partire dai suoi attributi. sigma_ab = (a, b) della purea."""
    conf = d['configurazione'].iloc[0]
    V = float(d['tensione'].iloc[0])
    sale = float(d['sale'].iloc[0])
    geo, occ = _geometria_prova(imp, conf)
    # fase di ciascuna sonda (per le temperature iniziali)
    in_polpetta = []
    for n in ('P1', 'P2', 'P3'):
        pos = np.array(geo['sonde'][n]).reshape(3, 1)
        in_polpetta.append(float(me.frazione_fase(pos, geo['centri'], geo['R'], 1e-3)[0]) > 0.5)
    T0 = np.array([d['T1'].iloc[0], d['T2'].iloc[0], d['T3'].iloc[0]], dtype=float)
    pur = [T0[i] for i in range(3) if not in_polpetta[i]]
    pol = [T0[i] for i in range(3) if in_polpetta[i]]
    T0_m = float(np.mean(pur)) if pur else float(np.mean(pol))
    T0_p = float(np.mean(pol)) if pol else T0_m
    comp = composizione_da_sale(sale, imp.burro, imp.acqua)
    p = dict(me.PARAMETRI_DEFAULT)
    p.update(
        matrice_a=float(sigma_ab[0]), matrice_b=float(sigma_ab[1]),
        comp_matrice=me.composizione_purea(**comp),
        prop_modo=imp.prop_modo, h_modo=imp.h_modo,
        V_rms=(V if imp.tensione_rms else V / np.sqrt(2.0)) * (1.0),
        T_iniziale=T0_m, T_iniziale_polpetta=T0_p,
        T_amb=imp.T_amb if imp.T_amb is not None else 21.0,
        durata=float(d['tempo'].max()),
    )
    return geo, p, in_polpetta


# ---------------------------------------------------------------------------
# Simulazione e metriche
# ---------------------------------------------------------------------------
_CACHE_MESH = {}


def mesh_per(geo, h_mm):
    chiave = (round(geo['Lx'], 6), round(geo['Ly'], 6), round(geo['Lz'], 6), tuple(round(c[0], 6) for c in geo['centri']),
              round(geo['R'], 6), round(h_mm, 4))
    if chiave not in _CACHE_MESH:
        if len(_CACHE_MESH) > 12:
            _CACHE_MESH.clear()
        _CACHE_MESH[chiave] = me.costruisci_mesh(geo, h_mm / 1000.0)
    return _CACHE_MESH[chiave]


def simula_prova(d, imp, sigma_ab, h_mm, dt, progress_callback=None):
    geo, p, in_pol = prepara_prova(d, imp, sigma_ab)
    p['dt'] = float(dt)
    mesh, basis, h_eff = mesh_per(geo, h_mm)
    ris = me.simula(geo, mesh, basis, h_eff, p, progress_callback=progress_callback)
    ris['in_polpetta'] = in_pol
    return ris


def confronta(ris, d):
    """Metriche per sonda: simulato interpolato agli istanti sperimentali."""
    t = d['tempo'].values.astype(float)
    out = []
    for i, n in enumerate(('P1', 'P2', 'P3')):
        exp = d[f'T{i + 1}'].values.astype(float)
        sim = np.interp(t, ris['tempo'], ris['T_sonde'][n])
        e = sim - exp
        ss_tot = np.sum((exp - exp.mean()) ** 2)
        out.append(dict(sonda=n, fase='polpetta' if ris['in_polpetta'][i] else 'purea',
                        RMSE=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                        errore_finale=float(e[-1]), errore_max=float(np.max(np.abs(e))),
                        salto_exp=float(exp[-1] - exp[0]), salto_sim=float(sim[-1] - sim[0]),
                        R2=float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else np.nan))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Calibrazione di sigma(T) della purea
# ---------------------------------------------------------------------------
def calibra_sigma(prove, imp, sigma0_ab, h_mm=5.0, dt=10.0, fit_alpha=False, solo_purea=True,
                  max_nfev=14, progress_callback=None):
    """Stima sigma(T) = s20 (1 + alpha (T-20)) della purea minimizzando lo scarto (simulato - misurato)
    sulle sonde delle prove date (lista di DataFrame di prova).
    Restituisce dict(s20, alpha, a, b, rmse, rmse_prima, n_simulazioni)."""
    alpha0, s20_0 = alpha_da_regressione(*sigma0_ab)
    pre = []
    for d in prove:
        geo, p, in_pol = prepara_prova(d, imp, sigma0_ab)
        sel = [i for i in range(3) if (not solo_purea) or (not in_pol[i])]
        if sel:
            pre.append((d, sel))
    if not pre:
        raise ValueError("Nessuna sonda in purea nelle prove selezionate: seleziona prove con almeno una sonda in purea "
                         "oppure disattiva 'solo sonde in purea'.")
    contatore = {'n': 0}

    def residui(x):
        s20 = float(np.exp(x[0]))
        alpha = float(x[1]) if fit_alpha else alpha0
        ab = sigma_da_s20_alpha(s20, alpha)
        res = []
        for d, sel in pre:
            ris = simula_prova(d, imp, ab, h_mm, dt)
            t = d['tempo'].values.astype(float)
            for i in sel:
                res.append(np.interp(t, ris['tempo'], ris['T_sonde'][f'P{i + 1}']) - d[f'T{i + 1}'].values)
        contatore['n'] += 1
        if progress_callback:
            progress_callback(contatore['n'])
        return np.concatenate(res)

    x0 = [np.log(s20_0)] + ([alpha0] if fit_alpha else [])
    lb = [np.log(0.03)] + ([0.0] if fit_alpha else [])
    ub = [np.log(20.0)] + ([0.10] if fit_alpha else [])
    r0 = residui(np.array(x0))
    sol = least_squares(residui, x0, bounds=(lb, ub), diff_step=0.03, max_nfev=max_nfev, x_scale=[1.0] + ([0.02] if fit_alpha else []))
    s20 = float(np.exp(sol.x[0]))
    alpha = float(sol.x[1]) if fit_alpha else alpha0
    a, b = sigma_da_s20_alpha(s20, alpha)
    return dict(s20=s20, alpha=alpha, a=a, b=b, rmse=float(np.sqrt(np.mean(sol.fun ** 2))),
                rmse_prima=float(np.sqrt(np.mean(r0 ** 2))), n_residui=int(len(r0)),
                n_chiamate=contatore['n'], s20_riferimento=s20_0, alpha_riferimento=alpha0)
