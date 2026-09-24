"""
Modelos de separação de gases por membranas (sistema binário A/B).

Base teórica: apostila PQI-3303 (Fenômenos de Transporte III), com os modelos de
Weller e Steiner (1950) e Walawender e Stern (1972).

Convenções
----------
x  : fração molar de A no lado de alta pressão (alimentação / concentrado)
y  : fração molar de A no lado de baixa pressão (permeado)
alfa : seletividade ideal  α* = P'_A / P'_B  (> 1, A é o componente mais permeável)
rho  : razão de pressões  pL / pH  (< 1)
theta: corte  θ = q_p / q_f

Unidades usadas para o dimensionamento da área:
    permeabilidade  P'  em Barrer  [1 Barrer = 1e-10 cm3(CNTP).cm/(s.cm2.cmHg)]
    espessura       t   em µm
    pressões        em bar (convertidas internamente para cmHg)
    vazão           q   em cm3(CNTP)/s
    área            Am  em m2
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

BAR_TO_CMHG = 75.0062
BARRER = 1e-10


# ---------------------------------------------------------------------------
# Relação de equilíbrio local (Eq. 21 / 55 / 99 da apostila)
# ---------------------------------------------------------------------------
def y_equilibrio(x, alfa, rho):
    """Composição local do permeado y em equilíbrio com o lado de alta pressão x.

    Resolve a equação quadrática (Eqs. 30-33):
        (1-α) y² + [r(1-x) + α r x - 1 + α] y - r α x = 0,   r = pH/pL = 1/rho
    """
    x = np.asarray(x, dtype=float)
    r = 1.0 / rho
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= 0.0:
            out[i] = 0.0
            continue
        if xi >= 1.0:
            out[i] = 1.0
            continue
        a = 1.0 - alfa
        b = r * (1.0 - xi) + alfa * r * xi - 1.0 + alfa
        c = -r * alfa * xi
        if abs(a) < 1e-12:  # α = 1  -> equação linear
            out[i] = -c / b
            continue
        disc = b * b - 4.0 * a * c
        raizes = [(-b + s * np.sqrt(disc)) / (2.0 * a) for s in (+1, -1)]
        # raiz fisicamente válida: 0 <= y <= 1 e pressão parcial de A maior do lado H (x >= rho*y)
        validas = [y for y in raizes if -1e-12 <= y <= 1.0 + 1e-12 and xi - rho * y >= -1e-12]
        out[i] = min(validas) if validas else np.nan
    return float(out[0]) if scalar else out


def y_max_ideal(x, alfa):
    """Limite de pureza quando pL/pH -> 0 (vácuo): y = α x / (1 + (α-1) x)."""
    x = np.asarray(x, dtype=float)
    return alfa * x / (1.0 + (alfa - 1.0) * x)


def x_min_concentrado(xf, alfa, rho):
    """Concentração mínima possível no concentrado (Eq. 37: θ -> 1, yp = xf).

    Resolve numericamente y_eq(xo) = xf.
    """
    try:
        return brentq(lambda x: y_equilibrio(x, alfa, rho) - xf, 1e-12, xf)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Parâmetros do módulo
# ---------------------------------------------------------------------------
@dataclass
class Params:
    xf: float          # fração molar de A na alimentação
    alfa: float        # seletividade α*
    pH: float          # bar
    pL: float          # bar
    PA: float          # Barrer  (permeabilidade de A)
    t: float           # µm      (espessura efetiva da película seletiva)
    qf: float          # cm3(CNTP)/s

    @property
    def rho(self):
        return self.pL / self.pH

    @property
    def permeancia_A(self):
        """P'_A / t  em cm3(CNTP)/(s.cm2.cmHg)."""
        return self.PA * BARRER / (self.t * 1e-4)

    @property
    def permeancia_B(self):
        return self.permeancia_A / self.alfa

    @property
    def pH_cmHg(self):
        return self.pH * BAR_TO_CMHG

    @property
    def pL_cmHg(self):
        return self.pL * BAR_TO_CMHG


def _cm2_to_m2(a):
    return a * 1e-4


# ---------------------------------------------------------------------------
# 1) Mistura perfeita (Weller & Steiner) - Caso 2: dados xf, θ, α*, pL/pH
# ---------------------------------------------------------------------------
def mistura_perfeita(p: Params, theta: float):
    """Retorna dict com xo, yp, Am (m2), recuperação de A no permeado."""
    xf, alfa, rho = p.xf, p.alfa, p.rho

    def x_do_balanco(y):
        return (xf - theta * y) / (1.0 - theta)  # Eq. 25

    def h(y):
        x = x_do_balanco(y)
        return y - y_equilibrio(max(x, 0.0), alfa, rho)

    y_hi = min(1.0, xf / theta)
    yp = brentq(h, xf, y_hi, xtol=1e-13)
    xo = x_do_balanco(yp)
    qf = p.qf
    Am = theta * qf * yp / (p.permeancia_A * (p.pH_cmHg * xo - p.pL_cmHg * yp))  # Eq. 29 (cm2)
    return dict(xo=xo, yp=yp, Am=_cm2_to_m2(Am), rec=theta * yp / xf)


# ---------------------------------------------------------------------------
# 2) Escoamento cruzado (Weller & Steiner, 1950) - integração numérica
#    Em cada ponto, y = y_eq(x) (permeado sem mistura).
#    dx/dθ* = (x - y)/(1 - θ*)      ;    dAm/dθ* = qf y / [ (P'A/t)(pH x - pL y) ]
# ---------------------------------------------------------------------------
def escoamento_cruzado(p: Params, theta: float, n_pontos=200):
    xf, alfa, rho = p.xf, p.alfa, p.rho
    perm = p.permeancia_A

    def f(ts, s):
        x, A = s
        x = max(x, 0.0)
        y = y_equilibrio(x, alfa, rho)
        dx = (x - y) / (1.0 - ts)
        dA = p.qf * y / (perm * (p.pH_cmHg * x - p.pL_cmHg * y))
        return [dx, dA]

    def x_zero(ts, s):
        return s[0] - 1e-9

    x_zero.terminal = True
    x_zero.direction = -1

    ts_eval = np.linspace(0.0, theta, n_pontos)
    sol = solve_ivp(f, (0.0, theta), [xf, 0.0], t_eval=ts_eval, events=x_zero,
                    rtol=1e-9, atol=1e-12, method="LSODA")
    if sol.status == 1:  # concentrado esgotou A antes de atingir θ
        return None
    ts = sol.t
    x = sol.y[0]
    A = sol.y[1]
    y = y_equilibrio(np.maximum(x, 0.0), alfa, rho)
    xo = x[-1]
    yp = (xf - (1.0 - theta) * xo) / theta  # Eq. 26 (balanço global)
    return dict(theta_star=ts, x=x, y=y, A=_cm2_to_m2(A), xo=xo, yp=yp,
                Am=_cm2_to_m2(A[-1]), rec=theta * yp / xf)


# ---------------------------------------------------------------------------
# 3) Contracorrente (Walawender & Stern, 1972)
#    Integra-se a partir da saída do concentrado (q' = 0, x = xo) rumo à alimentação.
#    Estados: q' (vazão de permeado acumulada), y (composição local do permeado).
#      x  = (qo xo + q' y)/(qo + q')
#      dq'/dA     = J_A + J_B
#      d(q'y)/dA  = J_A            ->   dy/dA = (J_A - y (J_A+J_B)) / q'
#    Chuta-se xo até que x = xf no ponto em que q' = θ qf.
# ---------------------------------------------------------------------------
def _contracorrente_dado_xo(p: Params, theta, xo, n_pontos=200):
    alfa, rho = p.alfa, p.rho
    qf = p.qf
    qo = (1.0 - theta) * qf
    qp_total = theta * qf
    PA, PB = p.permeancia_A, p.permeancia_B
    pH, pL = p.pH_cmHg, p.pL_cmHg

    def fluxos(x, y):
        JA = PA * (pH * x - pL * y)
        JB = PB * (pH * (1.0 - x) - pL * (1.0 - y))
        return JA, JB

    def f(A, s):
        qpp, y = s
        x = (qo * xo + qpp * y) / (qo + qpp)
        JA, JB = fluxos(x, y)
        dq = JA + JB
        dy = (JA - y * dq) / qpp
        return [dq, dy]

    def fim(A, s):
        return s[0] - qp_total

    fim.terminal = True
    fim.direction = 1

    eps = 1e-6 * qp_total
    y0 = y_equilibrio(xo, alfa, rho)
    sol = solve_ivp(f, (0.0, 1e12), [eps, y0], events=fim, rtol=1e-9, atol=1e-14,
                    method="LSODA", dense_output=True)
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        return None
    A_end = sol.t_events[0][0]
    qpp_end, y_end = sol.y_events[0][0]
    x_end = (qo * xo + qpp_end * y_end) / (qo + qpp_end)
    return dict(sol=sol, A_end=A_end, x_end=x_end, y_end=y_end, qo=qo)


def contracorrente(p: Params, theta: float, n_pontos=200):
    xf = p.xf

    def resid(xo):
        r = _contracorrente_dado_xo(p, theta, xo)
        return np.nan if r is None else r["x_end"] - xf

    xs = np.linspace(1e-6, xf, 60)
    rs = [resid(x) for x in xs]
    raiz = None
    for i in range(len(xs) - 1):
        if np.isfinite(rs[i]) and np.isfinite(rs[i + 1]) and rs[i] * rs[i + 1] < 0:
            raiz = brentq(resid, xs[i], xs[i + 1], xtol=1e-12)
            break
    if raiz is None:
        return None
    r = _contracorrente_dado_xo(p, theta, raiz)
    A_grid = np.linspace(0.0, r["A_end"], n_pontos)
    S = r["sol"].sol(A_grid)
    qpp, y = S[0], S[1]
    x = (r["qo"] * raiz + qpp * y) / (r["qo"] + qpp)
    # Convenção: posição medida a partir da entrada da alimentação (A = 0 na alimentação)
    A_pos = _cm2_to_m2(r["A_end"] - A_grid)
    order = np.argsort(A_pos)
    yp = y[-1]  # permeado sai no lado da alimentação (contracorrente)
    return dict(A_pos=A_pos[order], x=x[order], y=y[order],
                xo=raiz, yp=yp, Am=_cm2_to_m2(r["A_end"]), rec=theta * yp / xf)


# ---------------------------------------------------------------------------
# Transporte local através da membrana (resistências em série)
#   Líquido (diálise): Eqs. 3-9 da apostila
#   Gás:               Eqs. 10-16 da apostila
# As funções aceitam escalares ou arrays NumPy (por exemplo, para variar a espessura L).
# ---------------------------------------------------------------------------
R_GAS = 0.08206  # m3.atm/(kmol.K)


def transporte_liquido(C1, C2, kc1, kc2, D, Kp, L_um):
    """Permeação líquida (diálise).

    C1, C2 : concentrações nos seios dos líquidos 1 e 2 (mol/m3)
    kc1, kc2 : coeficientes de transferência de massa dos filmes (m/s)
    D : difusividade do soluto na membrana (m2/s)
    Kp : coeficiente de distribuição de equilíbrio K' (adimensional)
    L_um : espessura da membrana (µm)
    """
    PM = D * Kp / (L_um * 1e-6)  # m/s  (Eq. 4)
    R1, Rm, R2 = 1.0 / kc1, 1.0 / PM, 1.0 / kc2  # s/m
    N = (C1 - C2) / (R1 + Rm + R2)  # mol/(m2.s)  (Eq. 9)
    C1i = C1 - N * R1
    C2i = C2 + N * R2
    return dict(N=N, PM=PM, R1=R1, Rm=Rm, R2=R2, C1i=C1i, C2i=C2i,
                perfil=(C1, C1i, Kp * C1i, Kp * C2i, C2i, C2))


def transporte_gas(p1, p2, T, kc1, kc2, D, S, L_um):
    """Permeação gasosa.

    p1, p2 : pressões parciais de A nos seios das fases 1 e 2 (atm)
    T : temperatura (K)
    kc1, kc2 : coeficientes de transferência de massa nas fases gasosas (m/s)
    D : difusividade de A na membrana (m2/s)
    S : solubilidade em m3(CNTP)/[atm.m3 de sólido]  ->  H = S/22,414 (Eqs. 10 e 12)
    L_um : espessura da membrana (µm)

    N em kmol/(m2.s). Resistências em m2.s.atm/kmol.
    """
    H = S / 22.414  # kmol/(m3.atm)
    PM = D * H  # kmol/(s.m.atm)  (Eq. 12)
    R1 = R_GAS * T / kc1
    Rm = (L_um * 1e-6) / PM
    R2 = R_GAS * T / kc2
    N = (p1 - p2) / (R1 + Rm + R2)  # Eq. 16
    p1i = p1 - N * R1
    p2i = p2 + N * R2
    return dict(N=N, PM=PM, H=H, R1=R1, Rm=Rm, R2=R2, p1i=p1i, p2i=p2i,
                perfil=(p1, p1i, p1i, p2i, p2i, p2))


def perfil_esquematico(valores, larguras=(0.3, 0.4, 0.3)):
    """Monta o perfil (x, y) em três regiões: filme 1 | membrana | filme 2.

    valores = (bulk1, interface1_fluido, interface1_membrana, interface2_membrana,
               interface2_fluido, bulk2). A escala horizontal é esquemática.
    """
    x1 = larguras[0]
    x2 = larguras[0] + larguras[1]
    return [0.0, x1, x1, x2, x2, 1.0], list(valores)
