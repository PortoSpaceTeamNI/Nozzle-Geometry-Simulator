# sizing_from_thrust.py
import math
from rocketcea.cea_obj import CEA_Obj, add_new_fuel   # SÓ daqui

# --- Definir combustível Paraffina como nos outros scripts ---
card_str = """
fuel C30H62  C 30 H 62  wt%=83.00
h,cal=-158348.0  t(k)=298.15  rho=0.775
"""

add_new_fuel('Paraffin', card_str)

# RocketCEA base (unidades internas: psia, etc.)
cea = CEA_Obj(
    propName='',
    oxName='N2O',
    fuelName='Paraffin'
)

g0 = 9.80665
BAR_TO_PSIA = 14.5037738

def get_cf_and_pexit(Pc_bar, OF, eps, Pamb_bar):
    """
    Devolve:
      - cf (CFamb, coeficiente de thrust à pressão ambiente) [adim]
      - Pexit_bar (pressão à saída) [bar]
    """

    # Converter bar -> psia para o CEA
    Pc_psia   = Pc_bar   * BAR_TO_PSIA
    Pamb_psia = Pamb_bar * BAR_TO_PSIA

    # get_PambCf devolve 3 valores: CF, CFamb, mode
    CF, CFamb, mode = cea.get_PambCf(
        Pamb=Pamb_psia,
        Pc=Pc_psia,
        MR=OF,
        eps=eps
    )

    # Relação Pc/Pe (independente da unidade, é só um ratio)
    PcOvPe = cea.get_PcOvPe(Pc=Pc_psia, MR=OF, eps=eps)

    # Pc_bar / Pexit_bar = PcOvPe  ->  Pexit_bar = Pc_bar / PcOvPe
    Pexit_bar = Pc_bar / PcOvPe

    # cf que interessa para thrust a Pamb é o CFamb
    cf = CFamb

    if cf is None or Pexit_bar is None:
        raise RuntimeError("Não consegui ler cf ou P_exit do CEA.")

    return cf, Pexit_bar


def find_eps_for_ideal_expansion(Pc_bar, OF, Pamb_bar):
    # varre eps de 1.0 a 15.0
    eps_candidates = [x / 10.0 for x in range(10, 151)]
    best_eps = None
    best_err = 1e9

    for eps in eps_candidates:
        cf, Pexit_bar = get_cf_and_pexit(Pc_bar, OF, eps, Pamb_bar)
        err = abs(Pexit_bar - Pamb_bar)
        if err < best_err:
            best_err = err
            best_eps = eps

    return best_eps


def size_nozzle_from_thrust(F_req_N, Pc_bar, OF, Pamb_bar):
    # 1) encontrar eps ideal (P_exit ~ P_amb)
    eps = find_eps_for_ideal_expansion(Pc_bar, OF, Pamb_bar)

    # 2) cf para esse eps
    cf, Pexit_bar = get_cf_and_pexit(Pc_bar, OF, eps, Pamb_bar)

    # 3) área de garganta (Pc em Pa)
    Pc_Pa = Pc_bar * 1e5
    At = F_req_N / (cf * Pc_Pa)

    # 4) raio de garganta
    rt = math.sqrt(At / math.pi)

    return rt, eps, cf, Pexit_bar


if __name__ == "__main__":
    # EXEMPLO FIXO PARA TESTE:
    F_req_N   = 3000.0   # 1.5 kN
    Pc_bar    = 30.0
    OF        = 6.0
    Pamb_bar  = 1.0      # nível do mar

    rt, eps, cf, Pexit_bar = size_nozzle_from_thrust(F_req_N, Pc_bar, OF, Pamb_bar)

    print(f"F_req       = {F_req_N:.1f} N")
    print(f"Pc          = {Pc_bar:.2f} bar")
    print(f"OF          = {OF:.2f}")
    print(f"P_amb       = {Pamb_bar:.2f} bar")
    print()
    print(f"rt          = {rt*1000:.2f} mm")
    print(f"eps (Ae/At) = {eps:.3f}")
    print(f"cf          = {cf:.4f}")
    print(f"P_exit      = {Pexit_bar:.3f} bar")
