"""
Separação de gases por membranas - modelos de módulo (app didático)

Executar:  streamlit run app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from modelos import (
    Params,
    contracorrente,
    escoamento_cruzado,
    mistura_perfeita,
    x_min_concentrado,
    y_equilibrio,
    y_max_ideal,
)

st.set_page_config(page_title="Modelos de membranas", page_icon="🧪", layout="wide")

COR_MP, COR_CR, COR_CC = "#1f77b4", "#ff7f0e", "#2ca02c"
NOMES = {"MP": "Mistura perfeita", "CR": "Escoamento cruzado", "CC": "Contracorrente"}
CORES = {"MP": COR_MP, "CR": COR_CR, "CC": COR_CC}

# ---------------------------------------------------------------------------
# Barra lateral: parâmetros comuns a todos os modelos
# ---------------------------------------------------------------------------
st.sidebar.header("Parâmetros")
st.sidebar.caption("Sistema binário A/B. A é o componente mais permeável.")

xf = st.sidebar.slider("Fração molar de A na alimentação, x_f", 0.02, 0.95, 0.20, 0.01)
alfa = st.sidebar.slider("Seletividade ideal, α* = P'A/P'B", 1.5, 100.0, 20.0, 0.5)
pH = st.sidebar.slider("Pressão do lado da alimentação, p_H (bar)", 1.0, 40.0, 10.0, 0.5)
razao = st.sidebar.slider("Razão de pressões, p_L/p_H", 0.01, 0.90, 0.10, 0.01)
theta = st.sidebar.slider("Corte, θ = q_p/q_f", 0.02, 0.90, 0.30, 0.01)

with st.sidebar.expander("Dimensionamento da área (opcional)", expanded=False):
    PA = st.number_input("Permeabilidade de A, P'A (Barrer)", 1.0, 10000.0, 100.0, 10.0)
    t_um = st.number_input("Espessura da película seletiva, t (µm)", 0.05, 100.0, 1.0, 0.05)
    qf = st.number_input("Vazão de alimentação, q_f (cm³(CNTP)/s)", 1.0, 1e6, 100.0, 10.0)

pL = razao * pH
p = Params(xf=xf, alfa=alfa, pH=pH, pL=pL, PA=PA, t=t_um, qf=qf)
st.sidebar.markdown(f"p_L = **{pL:.2f} bar**")

st.title("Separação de gases por membranas")
st.caption(
    "Modelos de módulo: mistura perfeita, escoamento cruzado e contracorrente "
    "(FT III - PQI 3303). Mova os controles da barra lateral e observe os gráficos."
)


# ---------------------------------------------------------------------------
# Cálculos (com cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calc_mp(p, th):
    return mistura_perfeita(p, th)


@st.cache_data(show_spinner=False)
def calc_cr(p, th):
    return escoamento_cruzado(p, th)


@st.cache_data(show_spinner=False)
def calc_cc(p, th):
    return contracorrente(p, th)


@st.cache_data(show_spinner="Calculando a varredura de θ nos três modelos...")
def varredura(p, thetas):
    linhas = []
    for th in thetas:
        r = {"theta": th}
        for k, fun in (("MP", mistura_perfeita), ("CR", escoamento_cruzado), ("CC", contracorrente)):
            try:
                res = fun(p, th)
            except Exception:
                res = None
            if res is None:
                r[k] = None
            else:
                r[k] = dict(yp=float(res["yp"]), xo=float(res["xo"]),
                            rec=float(res["rec"]), Am=float(res["Am"]))
        linhas.append(r)
    return linhas


def curva_equilibrio_fig(extra_traces=None, titulo="Curva de equilíbrio local"):
    xs = np.linspace(0, 1, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=xs, name="y = x (sem separação)",
                             line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=xs, y=y_max_ideal(xs, alfa), name="Limite ideal (p_L/p_H → 0)",
                             line=dict(color="#aaaaaa", dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=y_equilibrio(xs, alfa, razao),
                             name=f"Equilíbrio (α*={alfa:g}, p_L/p_H={razao:g})",
                             line=dict(color="black", width=3)))
    for tr in extra_traces or []:
        fig.add_trace(tr)
    fig.update_layout(title=titulo, xaxis_title="x  (A no lado de alta pressão)",
                      yaxis_title="y  (A no permeado)", height=480,
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def metricas(res, com_area=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Permeado, y_p", f"{res['yp']:.3f}")
    c2.metric("Concentrado, x_o", f"{res['xo']:.3f}")
    c3.metric("Recuperação de A no permeado", f"{100 * res['rec']:.1f} %")
    if com_area:
        c4.metric("Área de membrana, A_m", f"{res['Am']:.3g} m²")


tab_eq, tab_mp, tab_cr, tab_cc, tab_cmp, tab_teo = st.tabs(
    ["1. Equilíbrio local", "2. Mistura perfeita", "3. Escoamento cruzado",
     "4. Contracorrente", "5. Comparação", "Teoria"]
)

# ---------------------------------------------------------------------------
# 1. Equilíbrio
# ---------------------------------------------------------------------------
with tab_eq:
    st.subheader("A base de todos os modelos")
    st.markdown(
        "Em qualquer ponto da membrana, a composição local do permeado **y** depende da composição "
        "do lado de alta pressão **x**, da seletividade **α\\*** e da razão de pressões **p_L/p_H**. "
        "Os três modelos usam esta mesma relação. **O que muda é como ela é aplicada ao longo do módulo.**"
    )
    st.latex(r"\frac{y}{1-y}=\alpha^*\,\frac{x-(p_L/p_H)\,y}{(1-x)-(p_L/p_H)(1-y)}")
    extra = [go.Scatter(x=[xf], y=[y_equilibrio(xf, alfa, razao)], mode="markers",
                        marker=dict(size=12, color="red"),
                        name=f"Alimentação: y = {y_equilibrio(xf, alfa, razao):.3f}")]
    fig = curva_equilibrio_fig(extra)
    outros = st.multiselect("Comparar com outros valores de α*", [2, 5, 10, 20, 50, 100], default=[])
    xs = np.linspace(0, 1, 300)
    for a in outros:
        fig.add_trace(go.Scatter(x=xs, y=y_equilibrio(xs, a, razao), name=f"α* = {a}",
                                 line=dict(width=1.5)))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Quanto maior α*, mais a curva se afasta da diagonal (mais seletiva a membrana). "
        "Quanto menor p_L/p_H, mais a curva se aproxima do limite ideal. "
        "Com p_L/p_H alto, a força motriz do permeado diminui e a separação piora."
    )

# ---------------------------------------------------------------------------
# 2. Mistura perfeita
# ---------------------------------------------------------------------------
with tab_mp:
    st.subheader("Mistura perfeita")
    st.markdown(
        "Os dois lados são **bem misturados**: o concentrado tem composição única x_o e o permeado, "
        "y_p. Adequado para **baixa recuperação** (θ pequeno). "
        "A solução é a interseção entre a **curva de equilíbrio** e a **reta de operação** do balanço "
        "de massa, x_f = (1-θ)x_o + θ y_p."
    )
    try:
        m = calc_mp(p, theta)
    except Exception as e:
        m = None
        st.error(f"Não foi possível resolver o modelo: {e}")
    if m:
        metricas(m)
        xs = np.linspace(0, 1, 100)
        yop = (xf - (1 - theta) * xs) / theta
        mask = (yop >= 0) & (yop <= 1.02)
        extra = [
            go.Scatter(x=xs[mask], y=yop[mask], name="Reta de operação (balanço de A)",
                       line=dict(color=COR_MP, width=2.5)),
            go.Scatter(x=[xf], y=[xf], mode="markers", marker=dict(size=11, color="gray"),
                       name="Alimentação (x_f)"),
            go.Scatter(x=[m["xo"]], y=[m["yp"]], mode="markers+text",
                       marker=dict(size=13, color="red"), text=["solução"], textposition="top right",
                       name=f"x_o = {m['xo']:.3f} ; y_p = {m['yp']:.3f}"),
        ]
        st.plotly_chart(curva_equilibrio_fig(extra, "Interseção: equilíbrio × balanço de massa"),
                        use_container_width=True)

        xmin = x_min_concentrado(xf, alfa, razao)
        st.caption(
            f"Limite de operação (Eq. 37): nenhuma concentração no concentrado abaixo de "
            f"x_oM = {xmin:.4f} é possível (θ → 1)."
        )

        st.markdown("##### Compromisso pureza × recuperação")
        ths = tuple(np.round(np.linspace(0.02, 0.90, 30), 3))
        rows = []
        for th in ths:
            try:
                rows.append((th, calc_mp(p, th)))
            except Exception:
                pass
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=[r[0] for r in rows], y=[r[1]["yp"] for r in rows],
                                 name="Pureza do permeado, y_p", line=dict(color=COR_MP)))
        fig.add_trace(go.Scatter(x=[r[0] for r in rows], y=[r[1]["rec"] for r in rows],
                                 name="Recuperação de A", line=dict(color="crimson")))
        fig.add_trace(go.Scatter(x=[r[0] for r in rows], y=[r[1]["Am"] for r in rows],
                                 name="Área A_m (m²)", line=dict(color="gray", dash="dot")),
                      secondary_y=True)
        fig.add_vline(x=theta, line_dash="dash", line_color="black")
        fig.update_xaxes(title_text="Corte θ")
        fig.update_yaxes(title_text="Fração", range=[0, 1.02], secondary_y=False)
        fig.update_yaxes(title_text="Área (m²)", secondary_y=True)
        fig.update_layout(height=430, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
        st.info("Aumentar θ recupera mais A, mas **dilui** o permeado (y_p cai) e exige mais área.")

# ---------------------------------------------------------------------------
# 3. Escoamento cruzado
# ---------------------------------------------------------------------------
with tab_cr:
    st.subheader("Escoamento cruzado")
    st.markdown(
        "O concentrado escoa em **plug flow** paralelo à membrana e o permeado sai "
        "perpendicularmente, **sem mistura**. Em cada ponto, y é dado pelo equilíbrio local com o x daquele "
        "ponto, então x cai ao longo do módulo. Aproxima o módulo espiral."
    )
    r = calc_cr(p, theta)
    if r is None:
        st.warning("Com este θ, o componente A se esgota no concentrado antes de o corte ser atingido. "
                   "Reduza θ ou α*.")
    else:
        metricas(r)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Composições ao longo da área", "Caminho no diagrama x-y"))
        fig.add_trace(go.Scatter(x=r["A"], y=r["x"], name="x (lado de alta pressão)",
                                 line=dict(color=COR_CR, width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r["A"], y=r["y"], name="y local (permeado instantâneo)",
                                 line=dict(color="black", dash="dash")), row=1, col=1)
        fig.add_hline(y=r["yp"], line_dash="dot", line_color="red", row=1, col=1,
                      annotation_text=f"y_p médio = {r['yp']:.3f}")
        xs = np.linspace(0, 1, 200)
        fig.add_trace(go.Scatter(x=xs, y=y_equilibrio(xs, alfa, razao), name="Equilíbrio",
                                 line=dict(color="lightgray")), row=1, col=2)
        fig.add_trace(go.Scatter(x=r["x"], y=r["y"], name="Trajetória", mode="lines",
                                 line=dict(color=COR_CR, width=3)), row=1, col=2)
        fig.update_xaxes(title_text="Área acumulada A (m²)", row=1, col=1)
        fig.update_xaxes(title_text="x", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="fração molar de A", range=[0, 1.02], row=1, col=1)
        fig.update_yaxes(title_text="y", range=[0, 1.02], row=1, col=2)
        fig.update_layout(height=450, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "A alimentação vê o permeado mais rico no início, quando x ainda é alto. "
            "O y_p médio é uma média ponderada desses valores locais. "
            "Por isso, o permeado é mais puro que no modelo de mistura perfeita "
            "(em que toda a membrana opera na composição baixa x_o)."
        )

# ---------------------------------------------------------------------------
# 4. Contracorrente
# ---------------------------------------------------------------------------
with tab_cc:
    st.subheader("Escoamento em contracorrente")
    st.markdown(
        "As duas correntes escoam em **plug flow** e em **sentidos opostos**. O permeado que sai pela "
        "extremidade da alimentação está em contato com o gás mais rico, o que maximiza a força motriz. "
        "O sistema de EDOs de Walawender e Stern é integrado a partir da saída do concentrado, "
        "com x_o ajustado por tentativa e erro até reproduzir x_f."
    )
    r = calc_cc(p, theta)
    if r is None:
        st.warning("Não foi encontrada solução para este conjunto de parâmetros. Tente reduzir θ.")
    else:
        metricas(r)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Composições ao longo do módulo", "Caminho no diagrama x-y"))
        fig.add_trace(go.Scatter(x=r["A_pos"], y=r["x"], name="x (lado de alta pressão)",
                                 line=dict(color=COR_CC, width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r["A_pos"], y=r["y"], name="y (lado do permeado)",
                                 line=dict(color="black", dash="dash")), row=1, col=1)
        xs = np.linspace(0, 1, 200)
        fig.add_trace(go.Scatter(x=xs, y=y_equilibrio(xs, alfa, razao), name="Equilíbrio",
                                 line=dict(color="lightgray")), row=1, col=2)
        fig.add_trace(go.Scatter(x=r["x"], y=r["y"], name="Trajetória", mode="lines",
                                 line=dict(color=COR_CC, width=3)), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=xs, name="y = x", line=dict(color="gray", dash="dot")),
                      row=1, col=2)
        fig.update_xaxes(title_text="Posição (área a partir da entrada da alimentação, m²)", row=1, col=1)
        fig.update_xaxes(title_text="x", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="fração molar de A", range=[0, 1.02], row=1, col=1)
        fig.update_yaxes(title_text="y", range=[0, 1.02], row=1, col=2)
        fig.update_layout(height=450, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Na contracorrente, o permeado **não precisa estar em equilíbrio local** com o concentrado: "
            "a trajetória fica abaixo da curva de equilíbrio, o que permite atingir maior pureza "
            "e menor área para o mesmo corte."
        )

# ---------------------------------------------------------------------------
# 5. Comparação
# ---------------------------------------------------------------------------
with tab_cmp:
    st.subheader("Comparação dos três modelos")
    st.caption(f"Mesmas condições: x_f = {xf}, α* = {alfa:g}, p_L/p_H = {razao:g}, θ = {theta}")

    resultados = {"MP": calc_mp(p, theta), "CR": calc_cr(p, theta), "CC": calc_cc(p, theta)}
    linhas = []
    for k, res in resultados.items():
        if res is None:
            linhas.append({"Modelo": NOMES[k], "y_p": "-", "x_o": "-", "Recuperação (%)": "-", "A_m (m²)": "-"})
        else:
            linhas.append({"Modelo": NOMES[k], "y_p": round(res["yp"], 4), "x_o": round(res["xo"], 4),
                           "Recuperação (%)": round(100 * res["rec"], 1), "A_m (m²)": float(f"{res['Am']:.4g}")})
    st.dataframe(linhas, use_container_width=True, hide_index=True)

    ths = tuple(np.round(np.linspace(0.05, 0.90, 18), 3))
    dados = varredura(p, ths)

    col1, col2 = st.columns(2)
    fig1 = go.Figure()
    fig2 = go.Figure()
    for k in ("MP", "CR", "CC"):
        pts = [(d["theta"], d[k]) for d in dados if d[k] is not None]
        if not pts:
            continue
        fig1.add_trace(go.Scatter(x=[q[1]["rec"] for q in pts], y=[q[1]["yp"] for q in pts],
                                  name=NOMES[k], line=dict(color=CORES[k], width=3)))
        fig2.add_trace(go.Scatter(x=[q[0] for q in pts], y=[q[1]["Am"] for q in pts],
                                  name=NOMES[k], line=dict(color=CORES[k], width=3)))
    fig1.update_layout(title="Pureza × recuperação", xaxis_title="Recuperação de A no permeado",
                       yaxis_title="Pureza do permeado, y_p", height=430,
                       xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02]),
                       legend=dict(orientation="h", y=-0.25))
    fig2.update_layout(title="Área de membrana × corte", xaxis_title="Corte θ",
                       yaxis_title="A_m (m²)", height=430, legend=dict(orientation="h", y=-0.25))
    fig2.add_vline(x=theta, line_dash="dash", line_color="black")
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    st.info(
        "Para a mesma recuperação, a **contracorrente** entrega o permeado mais puro, seguida pelo "
        "**escoamento cruzado** e, por último, pela **mistura perfeita**. Para o mesmo corte, "
        "a mistura perfeita exige mais área. A diferença cresce com θ e com α*. "
        "Com baixa recuperação, os três modelos praticamente coincidem."
    )

# ---------------------------------------------------------------------------
# Teoria
# ---------------------------------------------------------------------------
with tab_teo:
    st.subheader("Resumo teórico")
    st.markdown("**Transporte local (resistências em série).** Para gases, o fluxo de A é")
    st.latex(r"N_A=\frac{p_{A1}-p_{A2}}{\dfrac{RT}{k_{c1}}+\dfrac{L}{P_M}+\dfrac{RT}{k_{c2}}}"
             r"\qquad P_M=D_{AB}H")
    st.markdown("Nos modelos de módulo, a resistência dos filmes gasosos é desprezada (a membrana controla) e o fluxo do componente A é")
    st.latex(r"\frac{q_A}{A_m}=\frac{P'_A}{t}\,(p_H x - p_L y)")
    st.markdown("**Hipóteses comuns:** isotérmico, queda de pressão desprezível, permeabilidades constantes, "
                "sistema binário.")
    st.markdown("**Definições:** corte θ = q_p/q_f ; seletividade α\\* = P'_A/P'_B ; "
                "balanço global x_f = (1-θ) x_o + θ y_p.")
    st.markdown("**Modelos (diferem no padrão de escoamento):**")
    st.markdown(
        "- **Mistura perfeita:** um único ponto (x_o, y_p) satisfaz equilíbrio e balanço. "
        "A área vem de A_m = θ q_f y_p / [(P'_A/t)(p_H x_o - p_L y_p)].\n"
        "- **Escoamento cruzado:** integra-se dx/dθ\\* = (x - y)/(1 - θ\\*), com y dado pelo equilíbrio local, "
        "e dA_m/dθ\\* = q_f y / [(P'_A/t)(p_H x - p_L y)].\n"
        "- **Contracorrente:** integram-se as EDOs acopladas a partir do concentrado (q' = 0, x = x_o) "
        "e x_o é ajustado até atingir x_f na entrada da alimentação."
    )
    st.markdown(
        "**Nota sobre a implementação.** Para o escoamento cruzado, o app integra as EDOs "
        "numericamente (em vez de usar a solução analítica da Eq. 56). O resultado é equivalente e "
        "mais fácil de visualizar. A extensão multicomponente não está incluída."
    )
