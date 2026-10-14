"""
Pagina: confronto modello fisico - dati sperimentali (file Excel caricato dall'utente).
"""

import io
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

import multifase_engine as me
import confronto_esperimenti as ce

st.set_page_config(page_title="Confronto con gli esperimenti", layout="wide")
st.title("Confronto modello fisico - dati sperimentali")
st.caption("Carica il file Excel delle prove: l'app traduce gli attributi di ciascuna prova (configurazione, tensione, "
           "concentrazione salina, temperature iniziali) nei parametri del modello, simula e confronta con le sonde.")

LARG = 560
COL_SONDE = ['tab:blue', 'tab:orange', 'tab:green']
S_NODO_PASSO = 6.5e-5

if 'ce' not in st.session_state:
    st.session_state.ce = {'calib': {}}
S = st.session_state.ce


# =============================================================================
# Caricamento
# =============================================================================
@st.cache_data(show_spinner=False)
def carica(contenuto: bytes):
    return ce.leggi_dati(io.BytesIO(contenuto))


file = st.file_uploader("File Excel con le prove (.xlsx)", type=['xlsx'], key="ce_file")
if file is None:
    st.info("Carica un file con le colonne: **Configurazione, Tensione, Concentrazione salina, Ttempo, "
            "Temperatura 1, Temperatura 2, Temperatura 3** (una riga per istante, più prove nello stesso foglio).")
    st.stop()
try:
    df, avvisi = carica(file.getvalue())
except Exception as e:
    st.error(f"Impossibile leggere il file: {e}")
    st.stop()
for a in avvisi:
    st.warning(a)
el = ce.elenco_prove(df)
configs = sorted(df['configurazione'].unique())
tensioni = sorted(df['tensione'].unique())
sali = sorted(df['sale'].unique())

# =============================================================================
# Barra laterale: impostazioni del modello
# =============================================================================
with st.sidebar:
    st.header("Impostazioni del modello")
    with st.expander("Cella e tensione", expanded=True):
        cella = st.selectbox("Cella", list(me.CELLE.keys()), key="ce_cella")
        Lx_mm = st.number_input("Distanza tra gli elettrodi [mm]", value=me.CELLE[cella]['Lx'] * 1000, min_value=10.0,
                                key=f"ce_Lx_{cella}")
        modo_V = st.selectbox("La tensione del file è", ["tensione efficace (rms)", "tensione di picco"], key="ce_modoV")
    with st.expander("Ricetta della purea", expanded=True):
        st.caption("Dalla Tabella 4.3 (RPFB): burro e acqua costanti, fiocchi = resto meno il sale (es. sale 2.9% → fiocchi 18.8%).")
        burro = st.number_input("Burro [% w/w]", value=5.8, min_value=0.0, key="ce_burro")
        acqua = st.number_input("Acqua [% w/w]", value=72.5, min_value=1.0, key="ce_acqua")
    with st.expander("Configurazioni → posizioni delle polpette", expanded=True):
        st.caption("Spunta le posizioni occupate da una polpetta. 'Middle' e 'Side' sono dedotte dalle temperature "
                   "iniziali (la sonda nella polpetta parte più fredda): **da confermare**.")
        occ_df = pd.DataFrame({'P1': [ce.occupazione_default(c)[0] for c in configs],
                               'P2': [ce.occupazione_default(c)[1] for c in configs],
                               'P3': [ce.occupazione_default(c)[2] for c in configs]}, index=configs)
        occ_edit = st.data_editor(occ_df, key="ce_occ", width='stretch')
    with st.expander("Scambio, proprietà e risoluzione"):
        T_amb = st.number_input("Temperatura ambiente [°C]", value=21.0, key="ce_Tamb")
        h_lbl = {'h(T): convezione naturale + irraggiamento': 'naturale_irraggiamento', 'h costante 10 W/m²/K': 'costante'}
        h_sel = st.selectbox("Scambio sulla superficie libera", list(h_lbl.keys()), key="ce_h")
        prop_lbl = {'Choi–Okos dipendenti da T': 'choi_okos_T', 'Choi–Okos costanti a 50 °C': 'choi_okos_Tref'}
        prop_sel = st.selectbox("Proprietà termofisiche", list(prop_lbl.keys()), key="ce_prop")
        h_run = st.slider("Passo di mesh per le simulazioni [mm]", 1.5, 10.0, 3.0, 0.1, key="ce_hrun",
                          help="Le temperature alle sonde sono già stabili (±0.2 °C) a 5 mm.")
        dt_run = st.selectbox("Passo temporale delle simulazioni [s]", [5.0, 10.0], key="ce_dtrun")
        h_cal = st.slider("Passo di mesh per calibrazione/validazione [mm]", 2.5, 10.0, 5.0, 0.1, key="ce_hcal")
        dt_cal = st.selectbox("Passo temporale per calibrazione/validazione [s]", [10.0, 5.0, 15.0], key="ce_dtcal")

imp = ce.Impostazioni(cella=cella, Lx_mm=Lx_mm, tensione_rms=(modo_V.startswith("tensione efficace")),
                      burro=burro, acqua=acqua, h_modo=h_lbl[h_sel], T_amb=T_amb, prop_modo=prop_lbl[prop_sel])
imp.occupazione = {c: tuple(bool(occ_edit.loc[c, p]) for p in ('P1', 'P2', 'P3')) for c in configs}


# =============================================================================
# Utilità e figure
# =============================================================================
def id_prova(c, V, s):
    return f"{c} | {V:g} V | sale {s:g}"


def chiave_sale(s):
    return round(float(s), 4)


def sigma_scelta(sale, modo, man=None):
    """(a, b, etichetta) della purea secondo la scelta dell'utente."""
    cid, a_ref, b_ref = ce.regressione_riferimento(sale, imp.acqua)
    if modo.startswith("Calibrata"):
        c = S['calib'][chiave_sale(sale)]
        return c['a'], c['b'], f"calibrata (σ20 = {c['s20']:.2f} S/m)"
    if modo.startswith("Manuale"):
        a, b = ce.sigma_da_s20_alpha(*man)
        return a, b, f"manuale (σ20 = {man[0]:.2f} S/m)"
    return a_ref, b_ref, f"letteratura, comp. {cid} (σ20 = {20 * a_ref + b_ref:.2f} S/m)"


def fig_curve_dati(lista):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for (d, etich) in lista:
        for i in range(3):
            ax.plot(d['tempo'], d[f'T{i + 1}'], color=COL_SONDE[i], alpha=0.8, lw=1.5,
                    label=f'{etich} - P{i + 1}' if len(lista) <= 3 else None)
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('T [°C]'); ax.grid(alpha=.3)
    if len(lista) <= 3:
        ax.legend(fontsize=8)
    ax.set_title('Temperature misurate')
    plt.tight_layout()
    return fig


def fig_sovrapposizione(ris, d, titolo):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, n in enumerate(('P1', 'P2', 'P3')):
        fase = 'polpetta' if ris['in_polpetta'][i] else 'purea'
        ax.plot(d['tempo'], d[f'T{i + 1}'], 'o', ms=3.5, color=COL_SONDE[i], alpha=0.7, label=f'{n} ({fase}) misurata')
        ax.plot(ris['tempo'], ris['T_sonde'][n], '-', color=COL_SONDE[i], lw=2, label=f'{n} simulata')
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('T [°C]'); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title(titolo, fontsize=10)
    plt.tight_layout()
    return fig


def fig_errori(ris, d):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    t = d['tempo'].values.astype(float)
    for i, n in enumerate(('P1', 'P2', 'P3')):
        e = np.interp(t, ris['tempo'], ris['T_sonde'][n]) - d[f'T{i + 1}'].values
        ax.plot(t, e, color=COL_SONDE[i], lw=2, label=n)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('simulata - misurata [°C]'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Errore del modello nel tempo')
    plt.tight_layout()
    return fig


def fig_parity(risultati):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    colori = {'purea': 'tab:orange', 'polpetta': 'tab:brown'}
    mostrati = set()
    for r in risultati:
        for i in range(3):
            f = r['fasi'][i]
            ax.plot(r['exp'][i], r['sim'][i], '.', ms=3, alpha=0.45, color=colori[f], label=f if f not in mostrati else None)
            mostrati.add(f)
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel('T misurata [°C]'); ax.set_ylabel('T simulata [°C]'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Simulata vs misurata (tutti i punti)')
    plt.tight_layout()
    return fig


def fig_rmse_config(tab):
    g = tab.groupby('configurazione')['RMSE_tot'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.bar(g.index, g.values, color='tab:purple', alpha=0.8)
    ax.set_ylabel('RMSE medio [°C]'); ax.grid(alpha=.3, axis='y'); ax.set_title('Errore per configurazione')
    plt.tight_layout()
    return fig


tab_dati, tab_sim, tab_cal, tab_val = st.tabs(["1. Dati", "2. Simula una prova", "3. Calibra σ(T) della purea", "4. Validazione"])

# =============================================================================
# 1. Dati
# =============================================================================
with tab_dati:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prove", len(el)); c2.metric("Configurazioni", len(configs))
    c3.metric("Tensioni [V]", ", ".join(f"{v:g}" for v in tensioni))
    c4.metric("Concentrazioni saline", ", ".join(f"{s:g}" for s in sali))
    st.dataframe(el[['id', 'punti', 't_max', 'dt', 'T_ini', 'T_fin_media']].rename(columns={
        'id': 'Prova', 'punti': 'Punti', 't_max': 'Durata [s]', 'dt': 'Passo [s]', 'T_ini': 'T iniziale media [°C]',
        'T_fin_media': 'T finale media [°C]'}).round(2), hide_index=True, height=300)
    sel = st.multiselect("Mostra le curve delle prove", list(el['id']), default=list(el['id'])[:1], key="ce_selcurve")
    if sel:
        lista = []
        for idp in sel:
            r = el[el['id'] == idp].iloc[0]
            lista.append((ce.estrai_prova(df, r['configurazione'], r['tensione'], r['sale']), idp))
        st.pyplot(fig_curve_dati(lista), width=LARG + 100)

# =============================================================================
# 2. Simula una prova
# =============================================================================
with tab_sim:
    s1, s2, s3 = st.columns(3)
    conf_s = s1.selectbox("Configurazione", configs, key="ce_conf_s")
    V_opz = sorted(el[el['configurazione'] == conf_s]['tensione'].unique())
    V_s = s2.selectbox("Tensione [V]", V_opz, key=f"ce_V_s_{conf_s}")
    sali_opz = sorted(el[(el['configurazione'] == conf_s) & (el['tensione'] == V_s)]['sale'].unique())
    sale_s = s3.selectbox("Concentrazione salina", sali_opz, key=f"ce_sale_s_{conf_s}_{V_s}")
    d_s = ce.estrai_prova(df, conf_s, V_s, sale_s)

    modi = ["Riferimento da letteratura"]
    if chiave_sale(sale_s) in S['calib']:
        modi.insert(0, "Calibrata sui dati (per questa concentrazione)")
    modi.append("Manuale")
    modo_sig = st.radio("Conducibilità σ(T) della purea", modi, horizontal=True,
                       key=f"ce_modosig_{sale_s}_{'cal' if chiave_sale(sale_s) in S['calib'] else 'rif'}")
    man = None
    if modo_sig.startswith("Manuale"):
        m1, m2 = st.columns(2)
        man = (m1.number_input("σ a 20 °C [S/m]", value=1.5, min_value=0.01, key="ce_man_s20"),
               m2.number_input("Coefficiente α [1/K]", value=0.03, min_value=0.0, max_value=0.1, format="%.4f", key="ce_man_al"))
    a_sig, b_sig, lbl_sig = sigma_scelta(sale_s, modo_sig, man)

    geo_s, p_s, in_pol = ce.prepara_prova(d_s, imp, (a_sig, b_sig))
    with st.expander("Come il modello è stato impostato a partire dagli attributi della prova", expanded=True):
        info = pd.DataFrame({
            'Sonda': ['P1', 'P2', 'P3'],
            'Fase': ['polpetta' if x else 'purea' for x in in_pol],
            'T iniziale misurata [°C]': [d_s['T1'].iloc[0], d_s['T2'].iloc[0], d_s['T3'].iloc[0]],
        })
        st.dataframe(info.round(2), hide_index=True)
        st.write(f"- **Configurazione** `{conf_s}` → polpette in {[f'P{i + 1}' for i in range(3) if imp.occupazione[conf_s][i]] or 'nessuna posizione'}")
        st.write(f"- **Tensione** {V_s:g} V → {p_s['V_rms']:.1f} V rms su {geo_s['Lx'] * 1000:.0f} mm "
                 f"= {p_s['V_rms'] / (geo_s['Lx'] * 100):.2f} V/cm")
        comp = ce.composizione_da_sale(sale_s, imp.burro, imp.acqua)
        st.write(f"- **Sale** {sale_s:g}% → ricetta: burro {comp['w_burro']:.2f}%, sale {comp['w_sale']:.2f}%, "
                 f"fiocchi {comp['w_fiocchi']:.2f}%, acqua {comp['w_acqua']:.2f}%")
        st.write(f"- **Temperature iniziali del modello**: purea {p_s['T_iniziale']:.1f} °C, polpette {p_s['T_iniziale_polpetta']:.1f} °C "
                 "(medie delle sonde nella rispettiva fase)")
        st.write(f"- **σ della purea**: {lbl_sig}, σ = {a_sig:.4f}·T + {b_sig:.4f};  polpette: Tabella 5.1")
        st.write(f"- **Durata** {p_s['durata']:.0f} s; temperatura ambiente {imp.T_amb:.0f} °C")

    n_est, _ = me.stima_nodi(geo_s, h_run / 1000)
    st.caption(f"Mesh: ~{n_est:,} nodi; tempo indicativo ~{n_est * S_NODO_PASSO * p_s['durata'] / dt_run:.0f} s.")
    if st.button("Simula questa prova", type="primary", key="ce_run"):
        barra = st.progress(0.0, text="Simulazione in corso...")
        ris = ce.simula_prova(d_s, imp, (a_sig, b_sig), h_run, dt_run,
                              progress_callback=lambda f: barra.progress(f, text=f"Simulazione in corso... {f * 100:.0f}%"))
        barra.empty()
        S['ultima'] = dict(ris=ris, d=d_s, metr=ce.confronta(ris, d_s), titolo=f"{id_prova(conf_s, V_s, sale_s)}  -  σ {lbl_sig}")
    if 'ultima' in S:
        u = S['ultima']
        g1, g2 = st.columns(2)
        g1.pyplot(fig_sovrapposizione(u['ris'], u['d'], u['titolo']), width=LARG + 120)
        g2.pyplot(fig_errori(u['ris'], u['d']), width=LARG)
        m = u['metr']
        st.dataframe(m.rename(columns={'sonda': 'Sonda', 'fase': 'Fase', 'RMSE': 'RMSE [°C]', 'bias': 'Bias medio [°C]',
                                       'errore_finale': 'Errore finale [°C]', 'errore_max': 'Errore max [°C]',
                                       'salto_exp': 'Salto misurato [°C]', 'salto_sim': 'Salto simulato [°C]'}).round(2),
                     hide_index=True)
        st.metric("RMSE complessivo", f"{np.sqrt((m['RMSE'] ** 2).mean()):.2f} °C")
        csv = io.StringIO(); csv.write("tempo_s,exp_P1,exp_P2,exp_P3,sim_P1,sim_P2,sim_P3\n")
        t_ = u['d']['tempo'].values.astype(float)
        simi = [np.interp(t_, u['ris']['tempo'], u['ris']['T_sonde'][n]) for n in ('P1', 'P2', 'P3')]
        for k, t in enumerate(t_):
            csv.write(f"{t:g}," + ",".join(f"{u['d'][f'T{i + 1}'].iloc[k]:.3f}" for i in range(3)) + "," +
                      ",".join(f"{simi[i][k]:.3f}" for i in range(3)) + "\n")
        st.download_button("Scarica il confronto (CSV)", csv.getvalue(), "confronto_prova.csv", "text/csv")

# =============================================================================
# 3. Calibrazione
# =============================================================================
with tab_cal:
    st.markdown("""
Il modello ha bisogno della conducibilità $\\sigma(T)$ della **purea**, che per queste ricette non è nelle Tabelle 5.1 e 6.1.
Qui viene stimata dai dati, per ciascuna concentrazione salina, minimizzando lo scarto tra temperature simulate e misurate
nelle sonde in purea. Parametrizzazione: $\\sigma(T)=\\sigma_{20}\\,[1+\\alpha\\,(T-20)]$.
""")
    k1, k2 = st.columns(2)
    sale_c = k1.selectbox("Concentrazione salina da calibrare", sali, index=min(3, len(sali) - 1), key="ce_sale_c")
    el_c = el[np.isclose(el['sale'], sale_c)]
    cid, a_ref, b_ref = ce.regressione_riferimento(sale_c, imp.acqua)
    al_ref, s20_ref = ce.alpha_da_regressione(a_ref, b_ref)
    k2.info(f"Riferimento di partenza: regressione della comp. {cid} (σ20 = {s20_ref:.2f} S/m, α = {al_ref:.4f} 1/K).")

    # prove con almeno una sonda in purea
    candidate = []
    for _, r in el_c.iterrows():
        _, _, in_c = ce.prepara_prova(ce.estrai_prova(df, r['configurazione'], r['tensione'], r['sale']), imp, (a_ref, b_ref))
        if not all(in_c):
            candidate.append(r['id'])
    V_cent = tensioni[len(tensioni) // 2]
    default_cal = [i for i in candidate if f"| {V_cent:g} V |" in i]
    sel_cal = st.multiselect("Prove usate per calibrare (solo quelle con almeno una sonda in purea)", candidate,
                             default=default_cal, key=f"ce_selcal_{sale_c}",
                             help="Conviene calibrare su una tensione e validare sulle altre.")
    c1_, c2_, c3_ = st.columns(3)
    fit_alpha = c1_.checkbox("Calibra anche α", value=False, key="ce_fitalpha",
                             help="Con α fisso (dalla regressione di riferimento) si stima un solo parametro: più robusto.")
    solo_p = c2_.checkbox("Solo sonde in purea", value=True, key="ce_soloP")
    nfev = c3_.number_input("Valutazioni massime", value=14, min_value=4, max_value=60, key="ce_nfev")
    n_par = 2 if fit_alpha else 1
    sim_tot = (nfev + 1) * (n_par + 1) * max(1, len(sel_cal))
    st.caption(f"Costo massimo: ~{sim_tot} simulazioni da ~{1.7 * (5.0 / h_cal) ** 2:.1f} s: circa {sim_tot * 1.7 * (5.0 / h_cal) ** 2 / 60:.1f} minuti (stima, dipende dal server).")

    if st.button("Calibra σ(T)", type="primary", key="ce_calibra", disabled=not sel_cal):
        prove = []
        for idp in sel_cal:
            r = el_c[el_c['id'] == idp].iloc[0]
            prove.append(ce.estrai_prova(df, r['configurazione'], r['tensione'], r['sale']))
        barra = st.progress(0.0, text="Calibrazione in corso...")
        tot = (nfev + 1) * (n_par + 1)
        t0 = time.time()
        try:
            res = ce.calibra_sigma(prove, imp, (a_ref, b_ref), h_mm=h_cal, dt=dt_cal, fit_alpha=fit_alpha,
                                   solo_purea=solo_p, max_nfev=int(nfev),
                                   progress_callback=lambda n: barra.progress(min(1.0, n / tot), text=f"Calibrazione: valutazione {n}"))
        except ValueError as e:
            barra.empty()
            st.error(str(e))
            st.stop()
        barra.empty()
        res['prove_usate'] = list(sel_cal)
        res['durata_s'] = time.time() - t0
        S['calib'][chiave_sale(sale_c)] = res
        st.rerun()
    cal = S['calib'].get(chiave_sale(sale_c))
    if cal:
        st.success(f"Calibrazione disponibile per sale {sale_c:g}% (eseguita in {cal['durata_s']:.0f} s, {cal['n_chiamate']} valutazioni).")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("σ20 calibrata", f"{cal['s20']:.2f} S/m", delta=f"{cal['s20'] / cal['s20_riferimento'] - 1:+.0%} vs letteratura")
        q2.metric("α", f"{cal['alpha']:.4f} 1/K")
        q3.metric("RMSE prima", f"{cal['rmse_prima']:.2f} °C")
        q4.metric("RMSE dopo", f"{cal['rmse']:.2f} °C")
        st.write(f"σ(T) = {cal['a']:.4f}·T + {cal['b']:.4f}   (usata su {len(cal['prove_usate'])} prove: {', '.join(cal['prove_usate'])})")
        st.caption("σ calibrata è una conducibilità *efficace*: assorbe anche gli errori residui del modello (scambio, proprietà, "
                   "posizione delle sonde). Va giudicata soprattutto dalla validazione sulle prove NON usate (scheda 4).")
    with st.expander("Tutte le calibrazioni disponibili"):
        if S['calib']:
            st.dataframe(pd.DataFrame([{'Sale [%]': k, 'σ20 [S/m]': v['s20'], 'α [1/K]': v['alpha'], 'a': v['a'], 'b': v['b'],
                                        'RMSE dopo [°C]': v['rmse'], 'Prove': len(v['prove_usate'])} for k, v in sorted(S['calib'].items())]).round(4),
                         hide_index=True)
        else:
            st.write("Nessuna calibrazione ancora eseguita.")

# =============================================================================
# 4. Validazione
# =============================================================================
with tab_val:
    st.markdown("Esegue **tutte le prove** della concentrazione scelta con la σ calibrata e le confronta con le misure. "
                "Le prove non usate per calibrare mostrano quanto il modello è davvero predittivo.")
    v1, v2 = st.columns(2)
    sale_v = v1.selectbox("Concentrazione salina", sali, index=min(3, len(sali) - 1), key="ce_sale_v")
    disp = [("Calibrata sui dati" if chiave_sale(sale_v) in S['calib'] else None), "Riferimento da letteratura"]
    disp = [x for x in disp if x]
    modo_v = v2.radio("Conducibilità della purea", disp, horizontal=True,
                   key=f"ce_modov_{sale_v}_{'cal' if chiave_sale(sale_v) in S['calib'] else 'rif'}")
    el_v = el[np.isclose(el['sale'], sale_v)]
    sel_v = st.multiselect("Prove da eseguire", list(el_v['id']), default=list(el_v['id']), key=f"ce_selv_{sale_v}")
    st.caption(f"~{len(sel_v)} simulazioni; circa {len(sel_v) * 1.7 * (5.0 / h_cal) ** 2 * 270 / dt_cal / 27:.0f} s.")
    if st.button("Esegui la validazione", type="primary", key="ce_valida", disabled=not sel_v):
        a_v, b_v, lbl_v = sigma_scelta(sale_v, modo_v)
        righe, dettagli = [], []
        barra = st.progress(0.0, text="Validazione in corso...")
        usate = set(S['calib'].get(chiave_sale(sale_v), {}).get('prove_usate', [])) if modo_v.startswith("Calibrata") else set()
        for k, idp in enumerate(sel_v):
            r = el_v[el_v['id'] == idp].iloc[0]
            d = ce.estrai_prova(df, r['configurazione'], r['tensione'], r['sale'])
            ris = ce.simula_prova(d, imp, (a_v, b_v), h_cal, dt_cal)
            m = ce.confronta(ris, d)
            t = d['tempo'].values.astype(float)
            dettagli.append(dict(fasi=[('polpetta' if x else 'purea') for x in ris['in_polpetta']],
                                 exp=[d[f'T{i + 1}'].values for i in range(3)],
                                 sim=[np.interp(t, ris['tempo'], ris['T_sonde'][f'P{i + 1}']) for i in range(3)]))
            pu, po = m[m['fase'] == 'purea'], m[m['fase'] == 'polpetta']
            righe.append({'Prova': idp, 'configurazione': r['configurazione'], 'Tensione [V]': r['tensione'],
                          'Usata per calibrare': idp in usate,
                          'RMSE_tot': float(np.sqrt((m['RMSE'] ** 2).mean())),
                          'RMSE purea': float(np.sqrt((pu['RMSE'] ** 2).mean())) if len(pu) else np.nan,
                          'RMSE polpette': float(np.sqrt((po['RMSE'] ** 2).mean())) if len(po) else np.nan,
                          'Bias medio': float(m['bias'].mean())})
            barra.progress((k + 1) / len(sel_v), text=f"Validazione: {k + 1}/{len(sel_v)}")
        barra.empty()
        S['val'] = dict(tab=pd.DataFrame(righe), dettagli=dettagli, etichetta=lbl_v, sale=sale_v)
    if 'val' in S:
        vv = S['val']
        st.write(f"**σ della purea**: {vv['etichetta']}  (sale {vv['sale']:g}%)")
        tabv = vv['tab']
        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("RMSE medio, tutte le prove", f"{tabv['RMSE_tot'].mean():.2f} °C")
        if tabv['Usata per calibrare'].any():
            mm2.metric("RMSE medio, prove usate", f"{tabv[tabv['Usata per calibrare']]['RMSE_tot'].mean():.2f} °C")
            if (~tabv['Usata per calibrare']).any():
                mm3.metric("RMSE medio, prove NON usate", f"{tabv[~tabv['Usata per calibrare']]['RMSE_tot'].mean():.2f} °C")
        st.dataframe(tabv.drop(columns=['configurazione']).round(2), hide_index=True)
        h1, h2 = st.columns(2)
        h1.pyplot(fig_parity(vv['dettagli']), width=LARG)
        h2.pyplot(fig_rmse_config(tabv.rename(columns={'configurazione': 'configurazione'})), width=LARG)
        st.download_button("Scarica la tabella (CSV)", tabv.to_csv(index=False), "validazione.csv", "text/csv")

with st.expander("Ipotesi e limiti"):
    st.markdown("""
- **Sonde P1-P3** corrispondono a Temperatura 1-3 del file e ai centri delle posizioni P1-P3 della cella (assunto).
- **Concentrazione salina** interpretata come % w/w di sale nella purea, nella famiglia di ricette RPFB (burro e acqua costanti).
- **Temperature iniziali**: dalle sonde, distinte per fase; ambiente fisso (modificabile). Le temperature iniziali del file variano molto tra le prove (7-26 °C).
- **σ delle polpette**: Tabella 5.1 (non calibrata); la calibrazione riguarda solo la purea.
- **σ calibrata** = conducibilità efficace: assorbe errori residui del modello. Giudicare dalla validazione sulle prove non usate.
- **Mesh a 5 mm** per calibrazione/validazione: le temperature alle sonde differiscono di ≤ 0.2 °C da quelle a 2.5 mm (verificato su due configurazioni).
- Il modello non include ebollizione, evaporazione interna né fluidodinamica.
""")
