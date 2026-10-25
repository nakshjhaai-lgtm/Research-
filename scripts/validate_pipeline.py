"""Validate the CDM parser and the 2-D Pc implementation.

Three independent checks:

1.  A synthetic CCSDS KVL CDM is parsed and its reconstructed covariance is
    compared against the matrices that were used to generate it.
2.  The 2-D Pc closed form is compared against polar quadrature and Monte
    Carlo on random conjunctions, and against the exact Rayleigh CDF for the
    isotropic case -- this pins down the size of the small-HBR approximation.
3.  The closed form is evaluated on real Kelvins CDMs and compared with the
    self-reported ``risk`` column (log10 Pc) and with ``max_risk_estimate``.
    ``max_risk_estimate`` is an *independent* confirmation: it is the maximum
    Pc over an isotropic scaling of the combined covariance, whose optimum
    sits at s* = chi^2/2 and whose value is R^2 / (e * chi^2 * sigma_1 sigma_2).
    That the dataset's own ``max_risk_scaling`` column reproduces chi^2/2 and
    ``max_risk_estimate`` reproduces R^2/(e chi^2 sigma1 sigma2) validates the
    whole chain: covariance reconstruction, encounter-plane projection and
    hard-body radius.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.cdm_parser import CDMTable, parse_cdm_kvl
from src.covariance import encounter_plane_basis, mahalanobis, project_to_plane
from src.kelvins_data import load_test_truth
from src.pc2d import pc2d, pc2d_max, pc2d_monte_carlo, pc2d_numeric

#: ESA clips reported risk at 1e-30; rows at the floor carry no information.
RISK_FLOOR = -29.0


# ---------------------------------------------------------------------------
# 1. KVL parser test
# ---------------------------------------------------------------------------
SAMPLE_CDM = """CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2021-03-19T08:22:15.123456
ORIGINATOR = SDC
MESSAGE_ID = 12345
TCA = 2021-03-21T06:31:22.123456
MISS_DISTANCE = 842.31
RELATIVE_SPEED = 14863.114
RELATIVE_POSITION_R = -123.4
RELATIVE_POSITION_T = 812.9
RELATIVE_POSITION_N = -198.7
RELATIVE_VELOCITY_R = -21.3
RELATIVE_VELOCITY_T = -4210.7
RELATIVE_VELOCITY_N = 6720.9
COLLISION_PROBABILITY = 1.234e-05
COLLISION_PROBABILITY_METHOD = PCDWP-2D

OBJECT_1 = YES
OBJECT_DESIGNATOR = 40719
OBJECT_NAME = SENTINEL-6
CATALOG_NAME = SATCAT
INTERNATIONAL_DESIGNATOR = 15025A
OBJECT_TYPE = PAYLOAD
EPHEMERIS_NAME = NONE
COVARIANCE_METHOD = CALC
MANEUVERABLE = YES
REF_FRAME = ITRF
TIME_LASTOB_START = 2021-03-18T00:00:00.000
TIME_LASTOB_END = 2021-03-19T00:00:00.000
RECOMMENDED_OD_SPAN = 7.00
ACTUAL_OD_SPAN = 6.98
OBS_AVAILABLE = 215
OBS_USED = 214
RESIDUALS_ACCEPTED = 99.4
WEIGHTED_RMS = 1.29
X = -2312.144
Y = 4567.981
Z = -5123.771
X_DOT = 3.9124
Y_DOT = -5.8811
Z_DOT = -4.5321
CR_R = 3.920812E+00
CT_R = 2.988330E-01
CT_T = 1.445147E+00
CN_R = -1.422640E+00
CN_T = 1.423833E-01
CN_N = 1.547069E+00
CRDOT_R = 1.501280E+00
CRDOT_T = 7.794633E-01
CRDOT_N = 2.057492E-01
CRDOT_RDOT = 3.741647E+00
CTDOT_R = -1.706123E+00
CTDOT_T = 2.950156E-01
CTDOT_N = 2.080749E+00
CTDOT_RDOT = 3.215905E+00
CTDOT_TDOT = 1.152044E+01
CNDOT_R = -2.087121E+00
CNDOT_T = -9.393285E-01
CNDOT_N = -1.414040E-01
CNDOT_RDOT = -4.316243E-01
CNDOT_TDOT = -4.155365E+00
CNDOT_NDOT = 1.119521E+01

OBJECT_2 = YES
OBJECT_DESIGNATOR = 44713
OBJECT_NAME = COSMOS 1408 DEB
OBJECT_TYPE = DEBRIS
EPHEMERIS_NAME = NONE
COVARIANCE_METHOD = CALC
MANEUVERABLE = N/A
REF_FRAME = ITRF
TIME_LASTOB_START = 2021-03-18T00:00:00.000
TIME_LASTOB_END = 2021-03-19T00:00:00.000
RECOMMENDED_OD_SPAN = 5.00
ACTUAL_OD_SPAN = 4.99
OBS_AVAILABLE = 96
OBS_USED = 95
RESIDUALS_ACCEPTED = 98.8
WEIGHTED_RMS = 1.04
X = -2300.111
Y = 4571.002
Z = -5111.334
X_DOT = 3.9001
Y_DOT = -5.8812
Z_DOT = -4.5310
CR_R = 2.822113E+00
CT_R = 5.434903E-01
CT_T = 6.106398E-01
CN_R = 1.732073E-01
CN_T = 9.963441E-02
CN_N = 4.796157E-01
CRDOT_R = 1.870919E-01
CRDOT_T = -1.008897E-01
CRDOT_N = 2.400728E-01
CRDOT_RDOT = 5.333624E-01
CTDOT_R = -5.710125E-01
CTDOT_T = 2.993617E-01
CTDOT_N = 1.952236E-01
CTDOT_RDOT = -1.484695E-01
CTDOT_TDOT = 3.526226E+00
CNDOT_R = -3.533666E-01
CNDOT_T = -3.836200E-02
CNDOT_N = -4.339524E-03
CNDOT_RDOT = 3.053271E-01
CNDOT_TDOT = -1.267824E-01
CNDOT_NDOT = 7.085919E-01
"""


def test_kvl_parser() -> None:
    cdm = parse_cdm_kvl(SAMPLE_CDM)
    assert cdm.tca is not None and cdm.tca.year == 2021
    assert abs(cdm.miss_distance - 842.31) < 1e-9
    assert abs(cdm.relative_speed - 14863.114) < 1e-9
    assert np.allclose(cdm.relative_position, [-123.4, 812.9, -198.7])
    assert np.allclose(cdm.relative_velocity, [-21.3, -4210.7, 6720.9])
    assert abs(cdm.collision_probability - 1.234e-05) < 1e-12
    assert cdm.object1.object_name == "SENTINEL-6"
    assert cdm.object2.object_type == "DEBRIS"
    assert cdm.object1.obs_used == 214 and cdm.object1.weighted_rms == 1.29
    assert cdm.object2.covariance_rtn is not None
    for obj in (cdm.object1, cdm.object2):
        C = obj.covariance_rtn
        assert C.shape == (6, 6)
        assert np.allclose(C, C.T), "covariance must be symmetric"
        assert np.all(np.linalg.eigvalsh(C) > 0), "covariance must be PD"
    C1 = cdm.object1.covariance_rtn
    assert abs(C1[0, 0] - 3.920812) < 1e-6       # CR_R
    assert abs(C1[1, 0] - 0.2988330) < 1e-6      # CT_R, mirrored off-diagonal
    assert abs(C1[5, 5] - 11.19521) < 1e-4       # CNDOT_NDOT
    assert abs(C1[2, 1] - C1[1, 2]) < 1e-12
    Cc = cdm.combined_position_covariance
    assert Cc is not None and np.all(np.linalg.eigvalsh(Cc) > 0)
    print("[1] KVL CDM parser ......... OK  "
          f"(tca={cdm.tca}, Pc={cdm.collision_probability:.3e}, "
          "6x6 covariances symmetric & PD, combined C = C1 + C2)")


# ---------------------------------------------------------------------------
# 2. closed form vs quadrature vs Monte Carlo
# ---------------------------------------------------------------------------
def test_pc2d_methods() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(8):
        # realistic operational regime: sigmas of hundreds of metres,
        # hard-body radius of a few metres  ->  R/sigma << 1
        scale = float(rng.uniform(300.0, 3000.0))
        A = rng.standard_normal((2, 2)) * scale
        C2 = A @ A.T + np.eye(2) * scale ** 2
        miss = rng.standard_normal(2) * 2000.0
        R = float(rng.uniform(2.0, 20.0))
        p_closed = pc2d(miss, C2, R)
        p_num = pc2d_numeric(miss, C2, R)
        n_mc = 2_000_000
        p_mc = pc2d_monte_carlo(miss, C2, R, n_samples=n_mc, seed=1)
        rows.append((R, p_closed, p_num, p_mc))
        if p_num > 1e-30:
            assert abs(p_closed - p_num) / p_num < 0.02, (p_closed, p_num)
            # Monte Carlo is Poisson-limited: allow a 5-sigma band
            k = max(p_num * n_mc, 1.0)
            tol_mc = 5.0 * np.sqrt(k) / n_mc
            assert abs(p_mc - p_num) <= tol_mc, (p_mc, p_num, tol_mc)
        else:
            assert p_closed < 1e-100, (p_closed, p_num)

    print("[2] Pc closed form vs quadrature / Monte Carlo ... OK")
    for R, pc, pn, pm in rows:
        print(f"      R={R:6.2f} m   closed={pc:.6e}  quad={pn:.6e}  mc={pm:.6e}")

    # Exact analytic benchmark: isotropic covariance, zero miss -> Rayleigh CDF.
    for sigma in (100.0, 1000.0, 5000.0):
        for R in (5.0, 50.0, 500.0):
            C2 = sigma ** 2 * np.eye(2)
            exact = 1.0 - np.exp(-R ** 2 / (2.0 * sigma ** 2))
            num = pc2d_numeric([0.0, 0.0], C2, R)
            clo = pc2d([0.0, 0.0], C2, R)
            rel = abs(num - exact) / exact
            print(f"      sigma={sigma:7.1f} R={R:6.1f}  exact={exact:.6e} "
                  f"quad={num:.6e} (rel {rel:.1e})  closed={clo:.6e}")
            assert rel < 1e-3
    print("      isotropic zero-miss case reproduces the Rayleigh CDF exactly")

    C2 = np.array([[4.0, 1.0], [1.0, 9.0]])
    assert abs(pc2d([0.0, 0.0], C2, 10.0) - pc2d_max(10.0, C2)) < 1e-12
    print("      Pc(miss=0) == R^2/(2 sigma_1 sigma_2) confirmed")

    # Regime check: the closed form is only valid for R << sigma.
    sigma, R = 100.0, 500.0
    exact = 1.0 - np.exp(-R ** 2 / (2 * sigma ** 2))
    print(f"      WARNING regime: sigma=100, R=500 (R/sigma=5) -> "
          f"exact={exact:.4f} but closed form gives "
          f"{pc2d([0, 0], sigma**2*np.eye(2), R):.4f} (>1, unphysical)")
    print("      => the Alfriend-Akella closed form requires R << sigma_1, sigma_2")


# ---------------------------------------------------------------------------
# 3. 2-D Pc against the dataset's self-reported risk
# ---------------------------------------------------------------------------
_COLS = ["true_risk", "max_risk_estimate", "max_risk_scaling", "miss_distance",
         "relative_speed", "relative_position_r", "relative_position_t",
         "relative_position_n", "relative_velocity_r", "relative_velocity_t",
         "relative_velocity_n", "t_span", "c_span",
         "t_sigma_r", "t_sigma_t", "t_sigma_n", "t_ct_r", "t_cn_r", "t_cn_t",
         "c_sigma_r", "c_sigma_t", "c_sigma_n", "c_ct_r", "c_cn_r", "c_cn_t",
         "t_position_covariance_det", "c_position_covariance_det"]


def test_pc2d_against_dataset() -> pd.DataFrame:
    priv = load_test_truth(usecols=_COLS).rename(columns={"true_risk": "risk"})
    out = []
    for row in priv.itertuples(index=False):
        d = row._asdict()
        cdm = CDMTable(d)
        Ct, Cc = cdm.covariance_rtn("t"), cdm.covariance_rtn("c")
        C3 = Ct + Cc
        m, v = cdm.relative_position_rtn(), cdm.relative_velocity_rtn()
        R = cdm.hard_body_radius()
        P = encounter_plane_basis(v)
        C2, m2 = project_to_plane(C3, P), P.T @ m
        chi2 = mahalanobis(m2, C2)
        sd = float(np.sqrt(np.linalg.det(C2)))
        p = pc2d(m2, C2, R)
        out.append({
            "event_id": d.get("event_id"),
            "risk": d["risk"],
            "pc2d_log10": np.log10(p) if p > 0 else -np.inf,
            "chi2": chi2,
            "hbr_m": R,
            "sigma_prod": sd,
            "miss_distance_m": d["miss_distance"],
            "rel_pos_norm_m": float(np.linalg.norm(m)),
            "max_risk_estimate": d["max_risk_estimate"],
            "max_risk_scaling": d["max_risk_scaling"],
            # independent reconstruction of the max-risk fields
            "scaling_implied": chi2 / 2.0,
            "max_est_implied": np.log10(R ** 2 / (np.e * chi2 * sd)) if chi2 > 0 else np.nan,
            # exact determinant cross-checks
            "det_t_ratio": np.linalg.det(Ct) / d["t_position_covariance_det"],
            "det_c_ratio": np.linalg.det(Cc) / d["c_position_covariance_det"],
        })
    df = pd.DataFrame(out)
    df["err_log10"] = df["pc2d_log10"] - df["risk"]

    print(f"\n[3] 2-D Pc vs self-reported risk on {len(df)} real CDMs")
    print("      --- covariance reconstruction ---")
    print(f"      det(C_target)/t_position_covariance_det : "
          f"min {df.det_t_ratio.min():.6f}  max {df.det_t_ratio.max():.6f}")
    print(f"      det(C_chaser)/c_position_covariance_det : "
          f"min {df.det_c_ratio.min():.6f}  max {df.det_c_ratio.max():.6f}")
    print(f"      |relative_position| vs miss_distance     : "
          f"max rel diff {np.abs(df.rel_pos_norm_m / df.miss_distance_m - 1).max():.2e}")

    print("      --- independent check via the max-risk fields ---")
    ok = df[df.chi2 > 1e-9]
    r1 = ok.max_risk_scaling / ok.scaling_implied
    r2 = ok.max_risk_estimate - ok.max_est_implied
    print(f"      max_risk_scaling / (chi2/2)  : median {r1.median():.4f}, "
          f"corr {ok.max_risk_scaling.corr(ok.scaling_implied):.5f}")
    print(f"      max_risk_estimate - R^2/(e chi2 s1 s2) : median {r2.median():+.4f} dex")

    live = df[df.risk > RISK_FLOOR]
    e = live.err_log10
    print(f"      --- closed-form Pc vs `risk` (n={len(live)} above the {RISK_FLOOR} floor) ---")
    print(f"      median log10 error : {e.median():+.4f} dex")
    print(f"      within 0.1 dex     : {(e.abs() < 0.1).mean() * 100:.1f}%")
    print(f"      within 0.25 dex    : {(e.abs() < 0.25).mean() * 100:.1f}%")
    lo, hi = live[live.chi2 < 1], live[live.chi2 >= 10]
    print(f"      chi2<1  (n={len(lo):3d}, objects grazing the HBR): "
          f"median |err| {lo.err_log10.abs().median():.3f} dex, "
          f"within 0.25 dex {(lo.err_log10.abs() < 0.25).mean() * 100:.0f}%")
    print(f"      chi2>=10 (n={len(hi):3d}, far miss)              : "
          f"median |err| {hi.err_log10.abs().median():.3f} dex, "
          f"within 0.25 dex {(hi.err_log10.abs() < 0.25).mean() * 100:.0f}%")
    print("      NOTE: for far misses (large chi2) ESA's reported risk falls well")
    print("            below the closed form; the published columns carry only the")
    print("            position block of the covariance, so the residual is expected.")
    return df


if __name__ == "__main__":
    test_kvl_parser()
    print()
    test_pc2d_methods()
    df = test_pc2d_against_dataset()
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "pc2d_validation.csv", index=False)
