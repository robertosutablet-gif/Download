from collections import defaultdict
import csv
import inspect
import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from core.network import ThermalNetwork, ThermalNode, ThermalBranch, HeatLoad
from solvers.static import solve_static

NODE_TYPE_OPTIONS = ["boundary", "stage", "thermal_mass"]
MATERIAL_OPTIONS = [
    "",
    "OFHC_Cu_RRR100",
    "Al6061_T6",
    "G10_Normal",
    "SS304",
    "Ti6Al4V",
    "Inconel718",
    "Inconel725",
    "Brass",
    "REBCO",
]
FIXED_TEMP_OPTIONS = ["", "300.0", "77.0", "48.0", "4.5", "4.2"]
BRANCH_TYPE_OPTIONS = ["conduction", "contact", "mli", "radiation"]
LOAD_CATEGORY_OPTIONS = ["joule_lead", "ac_loss", "eddy_current", "radiation_leak", "static_load", "custom"]
BOOL_OPTIONS = ["True", "False"]

PARAM_TEMPLATES = [
    json.dumps({"material": "OFHC_Cu_RRR100", "area_m2": 5e-4, "length_m": 0.04}),
    json.dumps({"material": "Brass", "area_m2": 5e-4, "length_m": 0.04}),
    json.dumps({"material": "Ti6Al4V", "area_m2": 4e-5, "length_m": 0.35}),
    json.dumps({"material": "REBCO", "area_m2": 1e-4, "length_m": 0.10}),
    json.dumps({"material": "G10_Normal", "area_m2": 4e-5, "length_m": 0.40}),
    json.dumps({"conductance_WK": 25.0}),
    json.dumps({"mode": "parametric_modified_lockheed", "area_m2": 0.65, "layers": 40, "layer_density_per_cm": 25.0, "residual_pressure_Pa": 1e-5}),
    json.dumps({"area_m2": 0.45, "emissivity_eff": 0.02, "view_factor": 1.0}),
]


# ====================================================================
# PATCH 1 — _StageCapacityCallable: separa w_elec dal warning
# ====================================================================
class _StageCapacityCallable:
    def __init__(self, stage_spec):
        self._stage = stage_spec
        self.last_electric_power_W: float = 0.0
        self.last_warning = None

    def _compute_pair(self, temp_K=None, var2=None, **_kwargs):
        if not self._stage.enabled:
            q_cap = 0.0
        elif temp_K is None:
            q_cap = float(self._stage.q_ref_W)
        else:
            q_cap = float(self._stage.capacity_at(temp_K))
        t_eval = float(self._stage.t_ref_K if temp_K is None else temp_K)
        t_amb = 300.0
        w_elec = 0.0
        if t_eval > 0.0 and t_amb > t_eval:
            carnot_cop = t_eval / (t_amb - t_eval)
            eff_cop = max(0.02, 0.15 * carnot_cop)
            q_ref = max(float(self._stage.q_ref_W), 1e-6)
            t_ref = max(float(self._stage.t_ref_K), 1.0)
            cop_ref = max(0.02, 0.15 * (t_ref / max(t_amb - t_ref, 1.0)))
            w_ref = q_ref / cop_ref
            w_elec = float(q_cap / eff_cop) if eff_cop > 0.0 else float(w_ref)
        self.last_electric_power_W = w_elec
        return float(q_cap), w_elec

    def __call__(self, temp_K=None, var2=None, **_kwargs):
        q_cap, _w_elec = self._compute_pair(temp_K=temp_K, var2=var2, **_kwargs)
        warning = None
        try:
            t = float(temp_K) if temp_K is not None else self._stage.t_ref_K
            t_grid = self._stage.t_grid_K
            if t_grid:
                t_min, t_max = min(t_grid), max(t_grid)
                if t < t_min - 1e-9 or t > t_max + 1e-9:
                    warning = (
                        f"{self._stage.stage_name}: T={t:.3f} K fuori mappa "
                        f"[{t_min:.2f}, {t_max:.2f}] K (clamped a np.interp)"
                    )
        except Exception:
            warning = None
        self.last_warning = warning
        return float(q_cap), warning

    def __float__(self):
        return float(self._stage.q_ref_W if self._stage.enabled else 0.0)

    def __repr__(self):
        return f"{float(self):.6g}"


# ====================================================================
# PATCH 2 — CryocoolerStageSpec: aggiunge mode, var2_kind, x_domain, y_domain
# ====================================================================
class CryocoolerStageSpec:
    def __init__(self, st_cfg: dict, scale: float = 1.0):
        self.raw = st_cfg
        self.scale = float(scale)
        self.enabled = bool(st_cfg.get("enabled", True))
        self.node_id = str(st_cfg.get("node_id", ""))
        self.t_ref_K = float(st_cfg.get("t_ref_K", 4.2))
        self.q_ref_W = float(st_cfg.get("q_ref_W", 1.0)) * self.scale
        self.t_grid_K = [float(x) for x in st_cfg.get("t_grid_K", [])]
        self.q_cap_W = [float(x) * self.scale for x in st_cfg.get("q_cap_W", [])]
        self.t_grid = list(self.t_grid_K)
        self.q_grid = list(self.q_cap_W)
        # Attributi di compatibilità con solvers/static.py (bound locali)
        self.mode = "1d_curve"
        self.var2_kind = "temperature"
        self.allow_extrapolation = False
        self.stage_name = str(st_cfg.get("stage_name", self.node_id or "stage"))
        self.capacity_W = _StageCapacityCallable(self)
        self.q_cap = self.capacity_W

    @property
    def x_domain(self):
        if not self.t_grid_K:
            return (float(self.t_ref_K), float(self.t_ref_K))
        return (min(self.t_grid_K), max(self.t_grid_K))

    @property
    def y_domain(self):
        return (0.0, 0.0)

    def capacity_at(self, temp_K: float) -> float:
        if not self.enabled:
            return 0.0
        if not self.t_grid_K or not self.q_cap_W:
            return float(self.q_ref_W)
        t_arr = np.asarray(self.t_grid_K, dtype=float)
        q_arr = np.asarray(self.q_cap_W, dtype=float)
        order = np.argsort(t_arr)
        return float(np.interp(float(temp_K), t_arr[order], q_arr[order]))

    def evaluate(self, temp_K: float, var2=None, **_kwargs):
        return self.capacity_W(temp_K, var2=var2, **_kwargs)

    def __iter__(self):
        yield np.asarray(self.t_grid_K, dtype=float)
        yield np.asarray(self.q_cap_W, dtype=float)

    def __len__(self):
        return 2

    def __getattr__(self, name: str):
        if name in self.raw:
            return self.raw[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def __getitem__(self, key):
        if isinstance(key, int):
            pair = (np.asarray(self.t_grid_K, dtype=float), np.asarray(self.q_cap_W, dtype=float))
            return pair[key]
        return self.raw[key]


class CryocoolerSpec:
    def __init__(self, cc_cfg: dict):
        self.raw = cc_cfg
        self.model = str(cc_cfg.get("model", "two_stage_map"))
        self.cooling_power_scale = float(cc_cfg.get("cooling_power_scale", 1.0))
        self.stage1 = CryocoolerStageSpec(cc_cfg.get("stage1", {}), scale=self.cooling_power_scale)
        self.stage2 = CryocoolerStageSpec(cc_cfg.get("stage2", {}), scale=self.cooling_power_scale)
        self.stage1_node = self.stage1.node_id or "stage1"
        self.stage2_node = self.stage2.node_id or "stage2"
        self.stage1_enabled = self.stage1.enabled
        self.stage2_enabled = self.stage2.enabled
        self.stage1_t_grid = list(self.stage1.t_grid_K)
        self.stage1_q_grid = list(self.stage1.q_cap_W)
        self.stage2_t_grid = list(self.stage2.t_grid_K)
        self.stage2_q_grid = list(self.stage2.q_cap_W)
        self.stage1_capacity_W = self.stage1.capacity_W
        self.stage2_capacity_W = self.stage2.capacity_W

    def stage1_capacity(self, temp_K: float) -> float:
        return self.stage1.capacity_at(temp_K)

    def stage2_capacity(self, temp_K: float) -> float:
        return self.stage2.capacity_at(temp_K)

    def __iter__(self):
        yield self.stage1
        yield self.stage2

    def __len__(self):
        return 2

    def __getattr__(self, name: str):
        if name in self.raw:
            return self.raw[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def __getitem__(self, key):
        if isinstance(key, int):
            return (self.stage1, self.stage2)[key]
        if key == "stage1":
            return self.stage1
        if key == "stage2":
            return self.stage2
        return self.raw[key]


MLI_MODE_OPTIONS = ["parametric_modified_lockheed", "lockheed", "table", "custom"]
CONTACT_PARAM_MODE_OPTIONS = ["conductance_WK", "resistance_KW"]
AREA_PRESET_OPTIONS = ["1e-5", "2e-5", "4e-5", "6e-5", "5e-4", "6e-4", "1.2e-3", "0.45", "0.65", "1.0"]
LENGTH_PRESET_OPTIONS = ["0.02", "0.04", "0.06", "0.10", "0.20", "0.35", "0.40", "0.45", "0.50"]
CONTACT_VAL_PRESET_OPTIONS = ["5.0", "10.0", "20.0", "25.0", "40.0", "100.0", "0.01", "0.05", "0.10"]
EMISSIVITY_PRESET_OPTIONS = ["0.01", "0.02", "0.03", "0.05", "0.10", "0.50", "0.85"]
VIEW_FACTOR_PRESET_OPTIONS = ["1.0", "0.8", "0.5", "0.25", "0.1"]
MLI_LAYERS_PRESET_OPTIONS = ["10", "20", "30", "40", "60", "80"]
MLI_DENSITY_PRESET_OPTIONS = ["15.0", "20.0", "25.0", "30.0", "40.0"]
MLI_PRESSURE_PRESET_OPTIONS = ["1e-6", "1e-5", "1e-4", "1e-3", "0.1"]


def _parse_num_or_default(raw_val, default_val, as_int=False):
    txt = str(raw_val).strip()
    if txt == "" or txt.lower() in ("none", "null"):
        return default_val
    num = float(txt)
    if not np.isfinite(num):
        raise ValueError(f"Valore numerico non valido: '{raw_val}'")
    return int(round(num)) if as_int else float(num)


def build_branch_params_from_guided(
    branch_type: str,
    cond_material: str,
    cond_area: str,
    cond_length: str,
    cnt_mode: str,
    cnt_value: str,
    rad_area: str,
    rad_emissivity: str,
    rad_view_factor: str,
    mli_mode: str,
    mli_area: str,
    mli_layers: str,
    mli_density: str,
    mli_pressure: str,
    fallback_params: dict | None = None,
) -> dict:
    btype = (branch_type or "conduction").strip()
    base_extra = {}
    if isinstance(fallback_params, dict):
        known_by_type = {
            "conduction": {"material", "area_m2", "length_m"},
            "contact": {"conductance_WK", "resistance_KW"},
            "radiation": {"area_m2", "emissivity_eff", "view_factor"},
            "mli": {"mode", "area_m2", "layers", "layer_density_per_cm", "residual_pressure_Pa"},
        }
        known = known_by_type.get(btype, set())
        base_extra = {k: v for k, v in fallback_params.items() if k not in known}

    if btype == "conduction":
        params = dict(base_extra)
        mat_s = (cond_material or "OFHC_Cu_RRR100").strip()
        if mat_s:
            params["material"] = mat_s
        params["area_m2"] = _parse_num_or_default(cond_area, 5e-4, as_int=False)
        params["length_m"] = _parse_num_or_default(cond_length, 0.04, as_int=False)
        return params

    if btype == "contact":
        params = dict(base_extra)
        mode_s = (cnt_mode or "conductance_WK").strip()
        if mode_s not in CONTACT_PARAM_MODE_OPTIONS:
            mode_s = "conductance_WK"
        params[mode_s] = _parse_num_or_default(cnt_value, 25.0, as_int=False)
        return params

    if btype == "radiation":
        params = dict(base_extra)
        params["area_m2"] = _parse_num_or_default(rad_area, 0.45, as_int=False)
        params["emissivity_eff"] = _parse_num_or_default(rad_emissivity, 0.02, as_int=False)
        params["view_factor"] = _parse_num_or_default(rad_view_factor, 1.0, as_int=False)
        return params

    if btype == "mli":
        params = dict(base_extra)
        params["mode"] = (mli_mode or "parametric_modified_lockheed").strip() or "parametric_modified_lockheed"
        params["area_m2"] = _parse_num_or_default(mli_area, 0.65, as_int=False)
        params["layers"] = _parse_num_or_default(mli_layers, 40, as_int=True)
        params["layer_density_per_cm"] = _parse_num_or_default(mli_density, 25.0, as_int=False)
        params["residual_pressure_Pa"] = _parse_num_or_default(mli_pressure, 1e-5, as_int=False)
        return params

    return dict(fallback_params) if isinstance(fallback_params, dict) else {}


def default_cryocooler_dict():
    return {
        "model": "two_stage_map",
        "cooling_power_scale": 1.0,
        "stage1": {
            "enabled": True,
            "node_id": "stage1",
            "t_ref_K": 48.0,
            "q_ref_W": 35.0,
            "t_grid_K": [20.0, 30.0, 40.0, 48.0, 60.0, 77.0],
            "q_cap_W": [8.0, 18.0, 28.0, 35.0, 46.0, 60.0],
        },
        "stage2": {
            "enabled": True,
            "node_id": "stage2",
            "t_ref_K": 4.2,
            "q_ref_W": 1.0,
            "t_grid_K": [2.5, 3.5, 4.2, 4.5, 6.0, 10.0],
            "q_cap_W": [0.25, 0.65, 1.00, 1.18, 2.10, 4.50],
        },
    }


def ensure_cryocooler_config(cfg: dict) -> dict:
    base = default_cryocooler_dict()
    cc = cfg.get("cryocooler")
    if not isinstance(cc, dict):
        cfg["cryocooler"] = base
        return cfg["cryocooler"]
    cc.setdefault("model", base["model"])
    cc.setdefault("cooling_power_scale", base["cooling_power_scale"])
    for st_key in ("stage1", "stage2"):
        st_base = base[st_key]
        st = cc.get(st_key)
        if not isinstance(st, dict):
            cc[st_key] = dict(st_base)
            continue
        st.setdefault("enabled", st_base["enabled"])
        st.setdefault("node_id", st_base["node_id"])
        st.setdefault("t_ref_K", st_base["t_ref_K"])
        st.setdefault("q_ref_W", st_base["q_ref_W"])
        t_arr = [float(x) for x in st.get("t_grid_K", st_base["t_grid_K"])]
        q_arr = [float(x) for x in st.get("q_cap_W", st_base["q_cap_W"])]
        if len(t_arr) != len(q_arr) or len(t_arr) == 0:
            t_arr = list(st_base["t_grid_K"])
            q_arr = list(st_base["q_cap_W"])
        pairs = sorted(zip(t_arr, q_arr), key=lambda p: p[0])
        st["t_grid_K"] = [p[0] for p in pairs]
        st["q_cap_W"] = [p[1] for p in pairs]
    return cc


def interpolate_stage_capacity(stage_cfg: dict, temp_K: float, scale: float = 1.0) -> float:
    t_grid = np.asarray(stage_cfg.get("t_grid_K", []), dtype=float)
    q_grid = np.asarray(stage_cfg.get("q_cap_W", []), dtype=float)
    if t_grid.size == 0 or q_grid.size == 0:
        return float(stage_cfg.get("q_ref_W", 0.0)) * float(scale)
    order = np.argsort(t_grid)
    t_sorted = t_grid[order]
    q_sorted = q_grid[order]
    q_val = float(np.interp(float(temp_K), t_sorted, q_sorted))
    return q_val * float(scale)


def default_config_dict():
    return {
        "network": {
            "name": "Criostato_Magnete_Criogenico",
            "ambient_temp_K": 300.0,
            "t_min_K": 3.5,
            "t_max_K": 300.0,
        },
        "cryocooler": default_cryocooler_dict(),
        "nodes": [
            {"id": "ambient",   "node_type": "boundary",     "material": "",               "mass_kg": 0.0,  "target_temp_K": 300.0, "fixed_temp_K": 300.0},
            {"id": "stage1",    "node_type": "stage",        "material": "OFHC_Cu_RRR100", "mass_kg": 2.5,  "target_temp_K": 48.0,  "fixed_temp_K": None},
            {"id": "anchor_s1", "node_type": "thermal_mass", "material": "OFHC_Cu_RRR100", "mass_kg": 0.4,  "target_temp_K": 50.0,  "fixed_temp_K": None},
            {"id": "stage2",    "node_type": "stage",        "material": "OFHC_Cu_RRR100", "mass_kg": 1.5,  "target_temp_K": 4.3,   "fixed_temp_K": None},
            {"id": "shield",    "node_type": "thermal_mass", "material": "Al6061_T6",      "mass_kg": 8.0,  "target_temp_K": 55.0,  "fixed_temp_K": None},
            {"id": "cold_mass", "node_type": "thermal_mass", "material": "OFHC_Cu_RRR100", "mass_kg": 35.0, "target_temp_K": 4.5,   "fixed_temp_K": None},
        ],
        "branches": [
            {"id": "cond_anchor_s1", "branch_type": "conduction", "from_node": "anchor_s1", "to_node": "stage1",
             "params": {"material": "OFHC_Cu_RRR100", "area_m2": 5e-4, "length_m": 0.04}},
            {"id": "cnt_anchor_s1",  "branch_type": "contact",    "from_node": "anchor_s1", "to_node": "stage1",
             "params": {"conductance_WK": 25.0}},
            {"id": "link_s1_shield", "branch_type": "conduction", "from_node": "stage1",    "to_node": "shield",
             "params": {"material": "OFHC_Cu_RRR100", "area_m2": 6e-4, "length_m": 0.10}},
            {"id": "cnt_s1_shield",  "branch_type": "contact",    "from_node": "stage1",    "to_node": "shield",
             "params": {"conductance_WK": 20.0}},
            {"id": "mli_amb_shield", "branch_type": "mli",        "from_node": "ambient",   "to_node": "shield",
             "params": {"mode": "parametric_modified_lockheed", "area_m2": 0.65, "layers": 40, "layer_density_per_cm": 25.0, "residual_pressure_Pa": 1e-5}},
            {"id": "rad_amb_shield", "branch_type": "radiation",  "from_node": "ambient",   "to_node": "shield",
             "params": {"area_m2": 0.65, "emissivity_eff": 0.03, "view_factor": 1.0}},
            {"id": "rad_shield_cm",  "branch_type": "radiation",  "from_node": "shield",    "to_node": "cold_mass",
             "params": {"area_m2": 0.45, "emissivity_eff": 0.02, "view_factor": 1.0}},
            {"id": "busbar_s2_cm",   "branch_type": "conduction", "from_node": "stage2",    "to_node": "cold_mass",
             "params": {"material": "OFHC_Cu_RRR100", "area_m2": 1.2e-3, "length_m": 0.06}},
            {"id": "cnt_busbar_cm",  "branch_type": "contact",    "from_node": "stage2",    "to_node": "cold_mass",
             "params": {"conductance_WK": 40.0}},
            {"id": "lead_amb_anc",   "branch_type": "conduction", "from_node": "ambient",   "to_node": "anchor_s1",
             "params": {"material": "G10_Normal", "area_m2": 4e-5, "length_m": 0.45}},
            {"id": "lead_anc_cm",    "branch_type": "conduction", "from_node": "anchor_s1", "to_node": "cold_mass",
             "params": {"material": "G10_Normal", "area_m2": 2e-5, "length_m": 0.40}},
            {"id": "strut_amb_anc",  "branch_type": "conduction", "from_node": "ambient",   "to_node": "anchor_s1",
             "params": {"material": "G10_Normal", "area_m2": 6e-5, "length_m": 0.35}},
            {"id": "strut_anc_cm",   "branch_type": "conduction", "from_node": "anchor_s1", "to_node": "cold_mass",
             "params": {"material": "G10_Normal", "area_m2": 4e-5, "length_m": 0.35}},
            {"id": "diag_anc_cm",    "branch_type": "conduction", "from_node": "anchor_s1", "to_node": "cold_mass",
             "params": {"material": "G10_Normal", "area_m2": 1e-5, "length_m": 0.40}},
        ],
        "loads": [
            {"id": "joule_lead_amb_anc", "node": "anchor_s1", "power_W": 0.15, "category": "joule_lead", "enabled": True},
            {"id": "joule_lead_anc_cm",  "node": "cold_mass", "power_W": 0.01, "category": "joule_lead", "enabled": True},
            {"id": "ac_losses_cm",       "node": "cold_mass", "power_W": 0.05, "category": "ac_loss",    "enabled": False},
        ],
    }


def build_network_from_config(cfg: dict) -> ThermalNetwork:
    cc_cfg = ensure_cryocooler_config(cfg)
    cc_obj = CryocoolerSpec(cc_cfg)
    net_cfg = cfg["network"]
    net = ThermalNetwork(
        name=str(net_cfg["name"]),
        ambient_temp_K=float(net_cfg["ambient_temp_K"]),
    )
    setattr(net, "cryocooler_config", cc_obj)
    setattr(net, "cryocooler", cc_obj)

    for nd in cfg["nodes"]:
        fixed_t = nd.get("fixed_temp_K", None)
        fixed_t_val = None if fixed_t in ("", None) else float(fixed_t)
        mat_val = nd.get("material", "")
        kwargs = {
            "id": str(nd["id"]),
            "node_type": str(nd["node_type"]),
            "target_temp_K": float(nd["target_temp_K"]),
            "fixed_temp_K": fixed_t_val,
        }
        if mat_val:
            kwargs["material"] = str(mat_val)
        mass_val = nd.get("mass_kg", None)
        if mass_val not in ("", None) and float(mass_val) > 0.0:
            kwargs["mass_kg"] = float(mass_val)
        node_obj = ThermalNode(**kwargs)
        if str(nd["id"]) == cc_obj.stage1_node:
            setattr(node_obj, "cryocooler_stage", cc_obj.stage1)
        elif str(nd["id"]) == cc_obj.stage2_node:
            setattr(node_obj, "cryocooler_stage", cc_obj.stage2)
        net.add_node(node_obj)

    for br in cfg["branches"]:
        net.add_branch(ThermalBranch(
            id=str(br["id"]),
            branch_type=str(br["branch_type"]),
            from_node=str(br["from_node"]),
            to_node=str(br["to_node"]),
            params=dict(br["params"]),
        ))

    for ld in cfg["loads"]:
        net.add_load(HeatLoad(
            id=str(ld["id"]),
            node=str(ld["node"]),
            power_W=float(ld["power_W"]),
            category=str(ld["category"]),
            enabled=bool(ld["enabled"]),
        ))

    return net


# ====================================================================
# PATCH 3 — run_solver_with_cryocooler: evaluate_balances(dict), pre-warnings
# ====================================================================
def run_solver_with_cryocooler(net: ThermalNetwork, cfg: dict):
    cc_cfg = ensure_cryocooler_config(cfg)
    cc_obj = CryocoolerSpec(cc_cfg)
    setattr(net, "cryocooler_config", cc_obj)
    setattr(net, "cryocooler", cc_obj)

    # Warning non bloccanti pre-solve (se network.py è patchato)
    pre_warnings = []
    try:
        if hasattr(net, "collect_warnings"):
            pre_warnings = list(net.collect_warnings())
    except Exception:
        pass

    t_bounds = (float(cfg["network"]["t_min_K"]), float(cfg["network"]["t_max_K"]))
    sig = inspect.signature(solve_static)
    kwargs = {}
    for param_name in ("cryocooler_config", "cryocooler", "cooler", "cryocooler_spec"):
        if param_name in sig.parameters:
            kwargs[param_name] = cc_obj
            break

    sol = solve_static(net, t_bounds_K=t_bounds, **kwargs)
    if sol.stage_summary is None:
        sol.stage_summary = {}

    # Diagnostica residui: passa il DICT delle temperature (non np.ndarray)
    try:
        res_W, _, _, _ = net.evaluate_balances(sol.temperatures_K)
        finite_res = [abs(v) for v in res_W.values() if np.isfinite(v)]
        sol.stage_summary["max_residual_W"] = (
            float(max(finite_res)) if finite_res else float("nan")
        )
    except Exception:
        sol.stage_summary["max_residual_W"] = float("nan")

    # Fallback per Q_cap se il solver non l'ha popolato
    scale = float(cc_cfg.get("cooling_power_scale", 1.0))
    for idx_s, st_key in enumerate(("stage1", "stage2"), start=1):
        st_cfg = cc_cfg[st_key]
        cap_key = f"Q{idx_s}_cap_W"
        if cap_key not in sol.stage_summary or sol.stage_summary[cap_key] is None:
            t_st = sol.temperatures_K.get(
                st_cfg.get("node_id", st_key), st_cfg.get("t_ref_K", 4.2)
            )
            sol.stage_summary[cap_key] = interpolate_stage_capacity(
                st_cfg, t_st, scale=scale
            )

    # Mergia i warning pre-solve con quelli del solver
    if pre_warnings:
        merged = list(dict.fromkeys((sol.warnings or []) + pre_warnings))
        sol.warnings = merged

    return sol


OPT_OBJECTIVE_OPTIONS = [
    "max_margin_s2",
    "min_cold_heat",
    "min_grad_stage_to_target",
    "combined_margin_grad",
]

OPT_OBJECTIVE_LABELS = {
    "max_margin_s2": "Massimizza margine Stage 2 (e Stage 1)",
    "min_cold_heat": "Minimizza potenza termica verso cold_mass / stage2",
    "min_grad_stage_to_target": "Minimizza gradienti termici (shield-stage1, cold_mass-stage2)",
    "combined_margin_grad": "Combinato: margini positivi + riduzione gradienti e carichi",
}


def get_sizable_conduction_branches(cfg: dict) -> list[str]:
    out = []
    for br in cfg.get("branches", []):
        if str(br.get("branch_type", "")).strip() == "conduction":
            p = br.get("params", {}) if isinstance(br.get("params"), dict) else {}
            if "area_m2" in p and "length_m" in p:
                out.append(str(br.get("id", "")))
    return out


# ====================================================================
# PATCH 5 — compute_network_metrics: margine "vero" (Q_cap_max - Q_load)
# ====================================================================
def compute_network_metrics(net: ThermalNetwork, sol) -> dict:
    ss = sol.stage_summary or {}
    q1_l = float(ss.get("Q1_load_W", np.nan))
    q1_c = float(ss.get("Q1_cap_W", np.nan))
    q2_l = float(ss.get("Q2_load_W", np.nan))
    q2_c = float(ss.get("Q2_cap_W", np.nan))
    # Margine "vero": capacità massima sulla curva - carico (invece di ~0 a convergenza)
    q1_cap_max = float(ss.get("Q1_cap_max_W", q1_c))
    q2_cap_max = float(ss.get("Q2_cap_max_W", q2_c))
    m1 = (q1_cap_max - q1_l) if (np.isfinite(q1_cap_max) and np.isfinite(q1_l)) else np.nan
    m2 = (q2_cap_max - q2_l) if (np.isfinite(q2_cap_max) and np.isfinite(q2_l)) else np.nan

    t_map = sol.temperatures_K or {}
    t_s1 = float(t_map.get("stage1", np.nan))
    t_sh = float(t_map.get("shield", np.nan))
    t_s2 = float(t_map.get("stage2", np.nan))
    t_cm = float(t_map.get("cold_mass", np.nan))

    grad_s1_sh = abs(t_sh - t_s1) if (np.isfinite(t_sh) and np.isfinite(t_s1)) else 0.0
    grad_s2_cm = abs(t_cm - t_s2) if (np.isfinite(t_cm) and np.isfinite(t_s2)) else 0.0

    q_to_cold_mass = 0.0
    q_to_stage2 = 0.0
    for r in (sol.branch_results or []):
        bid = str(r.get("id", ""))
        q_val = float(r.get("Q_from_to_W", 0.0))
        br_obj = next((b for b in net.branches if b.id == bid), None)
        if br_obj is not None:
            if br_obj.to_node == "cold_mass":
                q_to_cold_mass += q_val
            elif br_obj.from_node == "cold_mass":
                q_to_cold_mass -= q_val
            if br_obj.to_node == "stage2":
                q_to_stage2 += q_val
            elif br_obj.from_node == "stage2":
                q_to_stage2 -= q_val

    return {
        "converged": bool(sol.converged),
        "Q1_load_W": q1_l,
        "Q1_cap_W": q1_c,
        "Q2_load_W": q2_l,
        "Q2_cap_W": q2_c,
        "Q1_cap_max_W": q1_cap_max,
        "Q2_cap_max_W": q2_cap_max,
        "margin1_W": m1,
        "margin2_W": m2,
        "T_stage1_K": t_s1,
        "T_shield_K": t_sh,
        "T_stage2_K": t_s2,
        "T_cold_mass_K": t_cm,
        "grad_stage1_shield_K": grad_s1_sh,
        "grad_stage2_cold_mass_K": grad_s2_cm,
        "Q_to_cold_mass_W": q_to_cold_mass,
        "Q_to_stage2_W": q_to_stage2,
    }


def run_optional_sizing_optimization(
    base_cfg: dict,
    branch_ids: list[str],
    opt_area: bool = True,
    opt_length: bool = True,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    objective_mode: str = "combined_margin_grad",
    max_iter: int = 25,
    Q1_load_max_W: float = 0.0,
    Q2_load_max_W: float = 0.0,
) -> tuple[dict, ThermalNetwork, object, dict]:
    from copy import deepcopy
    from scipy.optimize import minimize

    selected_ids = [bid for bid in branch_ids if bid]
    if not selected_ids:
        selected_ids = get_sizable_conduction_branches(base_cfg)
    if not selected_ids:
        raise ValueError("Nessun ramo di conduzione con area_m2 e length_m selezionato per il dimensionamento.")
    if not opt_area and not opt_length:
        raise ValueError("Seleziona almeno una grandezza da ottimizzare (sezione area_m2 e/o lunghezza length_m).")
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min >= scale_max:
        raise ValueError("Intervallo fattori di scala non valido (richiesto 0 < min < max).")

    slots = []
    for br in base_cfg.get("branches", []):
        bid = str(br.get("id", ""))
        if bid in selected_ids and str(br.get("branch_type", "")).strip() == "conduction":
            p = br.get("params", {}) if isinstance(br.get("params"), dict) else {}
            if opt_area and "area_m2" in p:
                slots.append((bid, "area_m2", float(p["area_m2"])))
            if opt_length and "length_m" in p:
                slots.append((bid, "length_m", float(p["length_m"])))

    if not slots:
        raise ValueError("Nessun parametro area_m2/length_m trovato nei rami selezionati.")

    def _eval_scales(x_vec):
        cand_cfg = deepcopy(base_cfg)
        br_by_id = {str(b.get("id", "")): b for b in cand_cfg.get("branches", [])}
        for val_scale, (bid, pkey, bval) in zip(x_vec, slots):
            if bid in br_by_id:
                br_by_id[bid]["params"][pkey] = float(bval * val_scale)
        net_c = build_network_from_config(cand_cfg)
        sol_c = run_solver_with_cryocooler(net_c, cand_cfg)
        met = compute_network_metrics(net_c, sol_c)
        return cand_cfg, net_c, sol_c, met

    def _loss_from_metrics(met: dict, x_vec) -> float:
        if not met["converged"]:
            return 1e4 + float(np.sum((x_vec - 1.0) ** 2))
        m1 = met["margin1_W"] if np.isfinite(met["margin1_W"]) else -10.0
        m2 = met["margin2_W"] if np.isfinite(met["margin2_W"]) else -10.0
        q_cm = met["Q_to_cold_mass_W"]
        q2_l = met["Q2_load_W"] if np.isfinite(met["Q2_load_W"]) else 0.0
        g1 = met["grad_stage1_shield_K"]
        g2 = met["grad_stage2_cold_mass_K"]

        margin_pen = 0.0
        if m2 < 0.0:
            margin_pen += 50.0 * abs(m2) + 5.0
        if m1 < 0.0:
            margin_pen += 20.0 * abs(m1) + 2.0
            if Q1_load_max_W > 0.0:
                q1_l_val = met.get("Q1_load_W", 0.0)
                if q1_l_val is not None and np.isfinite(q1_l_val):
                    excess1 = q1_l_val - Q1_load_max_W
                    if excess1 > 0.0:
                        margin_pen += 100.0 * excess1 + 10.0
            if Q2_load_max_W > 0.0:
                q2_l_val = met.get("Q2_load_W", 0.0)
                if q2_l_val is not None and np.isfinite(q2_l_val):
                    excess2 = q2_l_val - Q2_load_max_W
                    if excess2 > 0.0:
                        margin_pen += 100.0 * excess2 + 10.0

        reg = 0.02 * float(np.mean((x_vec - 1.0) ** 2))

        if objective_mode == "max_margin_s2":
            return (-2.0 * m2 - 0.25 * m1) + margin_pen + reg
        if objective_mode == "min_cold_heat":
            return (10.0 * max(q_cm, 0.0) + 5.0 * max(q2_l, 0.0) - 0.5 * m2) + margin_pen + reg
        if objective_mode == "min_grad_stage_to_target":
            return (0.2 * g2 + 0.05 * g1 - 0.5 * m2) + margin_pen + reg
        return (-1.5 * m2 - 0.2 * m1 + 2.0 * max(q_cm, 0.0) + 0.1 * g2 + 0.02 * g1) + margin_pen + reg

    best_state = {"cfg": None, "net": None, "sol": None, "met": None, "loss": np.inf}
    x0 = np.ones(len(slots), dtype=float)

    def _obj(x_vec):
        cand_cfg, net_c, sol_c, met = _eval_scales(x_vec)
        val = _loss_from_metrics(met, x_vec)
        if np.isfinite(val) and val < best_state["loss"]:
            best_state.update({"cfg": cand_cfg, "net": net_c, "sol": sol_c, "met": met, "loss": float(val)})
        return float(val)

    _obj(x0)
    bounds = [(float(scale_min), float(scale_max))] * len(slots)
    minimize(
        _obj,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-4, "xtol": 1e-3},
    )

    if best_state["cfg"] is None:
        best_cfg, best_net, best_sol, best_met = _eval_scales(x0)
    else:
        best_cfg, best_net, best_sol, best_met = (
            best_state["cfg"],
            best_state["net"],
            best_state["sol"],
            best_state["met"],
        )

    summary = {
        "n_vars": len(slots),
        "slots": [(b, k, round(v, 6)) for (b, k, v) in slots],
        "loss": best_state["loss"],
        "metrics": best_met,
    }
    return best_cfg, best_net, best_sol, summary


def quadratic_bezier_points(p0, p1, p2, n=150):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0_a, p1_a, p2_a = map(lambda v: np.asarray(v, dtype=float), (p0, p1, p2))
    return (1.0 - t) ** 2 * p0_a + 2.0 * (1.0 - t) * t * p1_a + t ** 2 * p2_a


def cubic_bezier_points(p0, c1, c2, p3, n=150):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0_a, c1_a, c2_a, p3_a = map(lambda v: np.asarray(v, dtype=float), (p0, c1, c2, p3))
    return ((1.0 - t) ** 3) * p0_a + 3.0 * ((1.0 - t) ** 2) * t * c1_a + 3.0 * (1.0 - t) * (t ** 2) * c2_a + (t ** 3) * p3_a


# ====================================================================
# PATCH 4 — compute_node_positions dinamico per nodi aggiunti via GUI
# ====================================================================
def compute_node_positions(net):
    """Posizioni per il grafo: hardcoded per nodi noti, arco circolare per
    nodi extra aggiunti via GUI.
    """
    hardcoded = {
        "anchor_s1": (0.14, 0.72),
        "stage1":    (0.50, 0.65),
        "shield":    (0.50, 0.44),
        "ambient":   (0.50, 0.17),
        "stage2":    (0.86, 0.72),
        "cold_mass": (0.86, 0.22),
    }
    node_ids = list(net.nodes.keys())
    positions = {nid: hardcoded[nid] for nid in node_ids if nid in hardcoded}
    extras = [nid for nid in node_ids if nid not in positions]
    if extras:
        import math
        n = len(extras)
        cx, cy, rx, ry = 0.5, 0.93, 0.45, 0.04
        for i, nid in enumerate(extras):
            angle = math.pi * (0.15 + 0.7 * (i / max(n - 1, 1)))
            positions[nid] = (cx + rx * math.cos(angle), cy + ry * math.sin(angle))
    return positions


def make_topology_figure(net, sol):
    pos = compute_node_positions(net)
    q_map = {r["id"]: r["Q_from_to_W"] for r in sol.branch_results}

    q_in_map = {nid: 0.0 for nid in net.nodes}
    q_out_map = {nid: 0.0 for nid in net.nodes}
    for b in net.branches:
        qv = float(q_map.get(b.id, 0.0))
        if not np.isfinite(qv):
            continue
        if qv >= 0.0:
            q_in_map[b.to_node] = q_in_map.get(b.to_node, 0.0) + qv
            q_out_map[b.from_node] = q_out_map.get(b.from_node, 0.0) + qv
        else:
            q_in_map[b.from_node] = q_in_map.get(b.from_node, 0.0) - qv
            q_out_map[b.to_node] = q_out_map.get(b.to_node, 0.0) - qv
    grouped = defaultdict(list)
    for b in net.branches:
        pair = tuple(sorted([b.from_node, b.to_node]))
        q_val = q_map.get(b.id, 0.0)
        if not grouped[pair]:
            grouped[pair].append((b.from_node, b.to_node, b.id, b.branch_type, q_val))
        else:
            ref_from, ref_to = grouped[pair][0][0], grouped[pair][0][1]
            signed_q = q_val if (b.from_node == ref_from and b.to_node == ref_to) else -q_val
            grouped[pair].append((b.from_node, b.to_node, b.id, b.branch_type, signed_q))

    cubic_ctrl = {tuple(sorted(["anchor_s1", "cold_mass"])): ((0.30, 0.87), (0.78, 0.87))}
    quad_ctrl = {tuple(sorted(["ambient", "anchor_s1"])): (0.22, 0.35)}
    pair_label_pos = {
        tuple(sorted(["anchor_s1", "stage1"])):    (0.32, 0.69),
        tuple(sorted(["shield", "stage1"])):       (0.50, 0.545),
        tuple(sorted(["ambient", "shield"])):      (0.50, 0.305),
        tuple(sorted(["ambient", "anchor_s1"])):   (0.26, 0.38),
        tuple(sorted(["anchor_s1", "cold_mass"])): (0.31, 0.82),
        tuple(sorted(["cold_mass", "shield"])):    (0.68, 0.33),
        tuple(sorted(["cold_mass", "stage2"])):    (0.86, 0.47),
    }

    fig, ax = plt.subplots(figsize=(15.5, 9.6))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title(f"Topologia Criostato: {net.name}", fontsize=13.0, fontweight="bold", pad=10)
    box_w, box_h = 0.135, 0.082

    for pair, items in grouped.items():
        n1, n2 = pair
        if n1 not in pos or n2 not in pos:
            continue
        ref_from, ref_to, _, _, _ = items[0]
        p0 = np.array(pos[ref_from], dtype=float)
        p2 = np.array(pos[ref_to], dtype=float)
        vec = p2 - p0
        dist = np.linalg.norm(vec)
        unit = vec / dist if dist > 1e-9 else np.array([1.0, 0.0])
        p0_trim = p0 + unit * 0.070
        p2_trim = p2 - unit * 0.070

        if pair in cubic_ctrl:
            c1, c2 = cubic_ctrl[pair]
            c1_a, c2_a = np.array(c1, dtype=float), np.array(c2, dtype=float)
            verts = [p0_trim, c1_a, c2_a, p2_trim]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
            ax.add_patch(FancyArrowPatch(path=MplPath(verts, codes), arrowstyle="-|>", mutation_scale=15, lw=2.2, color="#1d4ed8", linestyle="-", zorder=2))
            pts = cubic_bezier_points(p0_trim, c1_a, c2_a, p2_trim, n=150)
        elif pair in quad_ctrl:
            p1 = np.array(quad_ctrl[pair], dtype=float)
            verts = [p0_trim, p1, p2_trim]
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
            ax.add_patch(FancyArrowPatch(path=MplPath(verts, codes), arrowstyle="-|>", mutation_scale=15, lw=2.2, color="#1d4ed8", linestyle="-", zorder=2))
            pts = quadratic_bezier_points(p0_trim, p1, p2_trim, n=150)
        else:
            ax.annotate("", xy=(p2_trim[0], p2_trim[1]), xytext=(p0_trim[0], p0_trim[1]), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1d4ed8", linestyle="-"), zorder=2)
            pts = np.linspace(p0_trim, p2_trim, 150)

        has_cond_contact = any(it[3] == "contact" for it in items) and any(it[3] == "conduction" for it in items)
        if has_cond_contact:
            i_start = int(0.10 * (len(pts) - 1))
            i_end = int(0.22 * (len(pts) - 1))
            seg = pts[i_start : i_end + 1]
            ax.plot(seg[:, 0], seg[:, 1], color="#15803d", lw=4.5, solid_capstyle="butt", zorder=3)
            ax.scatter([seg[-1, 0]], [seg[-1, 1]], s=85, marker="s", facecolor="#dcfce7", edgecolor="#15803d", linewidth=1.6, zorder=4)

        lines = [f"{ref_from} -> {ref_to}:"]
        for _, _, bid, btype, qv in items:
            if btype == "contact":
                lines.append(f"• {bid} [contatto inizio conduzione]: {qv:+.4f} W")
            else:
                lines.append(f"• {bid} ({btype}): {qv:+.4f} W")

        lx, ly = pair_label_pos.get(pair, (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1])))
        ax.text(lx, ly, "\n".join(lines), fontsize=7.0, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.22", fc="#ffffff", ec="#1d4ed8", lw=1.1, alpha=0.97), zorder=5)

    for nid, (x, y) in pos.items():
        t_val = sol.temperatures_K.get(nid, 0.0)
        q_load = sum(ld.power_W for ld in net.loads if ld.enabled and ld.node == nid)
        q_through = max(q_in_map.get(nid, 0.0), q_out_map.get(nid, 0.0))
        q_sum = 0.0 if nid == "ambient" else q_load + q_through
        fc = "#fef3c7" if nid in ("stage1", "stage2") else ("#e2e8f0" if nid == "ambient" else "#eff6ff")
        ec = "#b45309" if nid in ("stage1", "stage2") else ("#475569" if nid == "ambient" else "#1e3a8a")
        ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=1.8, zorder=4))
        # Box più alto per accomodare nome + T + Q (3 righe)
        _has_q = q_sum > 0.0
        _box_h_local = 0.100 if _has_q else 0.082
        # Ridisegna il box con l'altezza corretta (sostituisce il patch generico precedente)
        ax.add_patch(FancyBboxPatch(
            (x - box_w / 2, y - _box_h_local / 2), box_w, _box_h_local,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            fc=fc, ec=ec, lw=1.8, zorder=4,
        ))
        # Riga 1: nome nodo
        ax.text(x, y + 0.026, nid, fontsize=9.0, fontweight="bold",
                ha="center", va="center", zorder=6)
        # Riga 2: temperatura (sempre visibile, gestisce nan)
        _t_txt = f"{t_val:.2f} K" if (t_val is not None and np.isfinite(t_val)) else "N/D"
        ax.text(x, y + 0.002, _t_txt, fontsize=7.6,
                ha="center", va="center", zorder=6)
        # Riga 3: potenza (solo se c'è un carico)
        if _has_q:
            ax.text(x, y - 0.024, f"Q = {q_sum:.2f} W", fontsize=6.8,
                    ha="center", va="center", color="#7a3e00", zorder=6)

    ss = sol.stage_summary or {}
    summary_text = (
        f"RIEPILOGO REGIME (Converged: {sol.converged}) | "
        f"Stage 1: {sol.temperatures_K.get('stage1', 0.0):.2f} K ({ss.get('Q1_load_W', 0.0):.3f}/{ss.get('Q1_cap_W', 0.0):.3f} W) | "
        f"Stage 2: {sol.temperatures_K.get('stage2', 0.0):.2f} K ({ss.get('Q2_load_W', 0.0):.3f}/{ss.get('Q2_cap_W', 0.0):.3f} W) | "
        f"anchor_s1: {sol.temperatures_K.get('anchor_s1', 0.0):.2f} K | "
        f"shield: {sol.temperatures_K.get('shield', 0.0):.2f} K | "
        f"cold_mass: {sol.temperatures_K.get('cold_mass', 0.0):.2f} K"
    )
    ax.text(0.50, 0.992, summary_text, fontsize=7.2, ha="center", va="top", bbox=dict(boxstyle="round,pad=0.22", fc="#f8fafc", ec="#475569", lw=1.2, alpha=0.98), zorder=7)

    legend_elements = [
        Line2D([0], [0], color="#1d4ed8", lw=2.2, linestyle="-", marker=">", label="Tratto blu: collegamento termico (conduzione / radiazione / MLI)"),
        Line2D([0], [0], color="#15803d", lw=4.5, linestyle="-", marker="s", markersize=8, markerfacecolor="#dcfce7", markeredgecolor="#15803d", label="Tratto verde all'inizio: resistenza/conduttanza di contatto in serie sulla conduzione"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=True, fontsize=8.5)
    fig.tight_layout()
    return fig


def export_results_report_doc(cfg: dict, net, sol, out_path: Path, opt_info: dict | None = None) -> Path:
    """Genera un documento Word (.docx) con i risultati della simulazione/ottimizzazione e la tabella completa dei rami."""
    out_path = Path(out_path)
    if out_path.suffix.lower() not in (".docx", ".doc"):
        out_path = out_path.with_suffix(".docx")

    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
        from docx.enum.section import WD_ORIENT
        from docx.oxml import OxmlElement, parse_xml
        from docx.oxml.ns import nsdecls, qn
    except ImportError as exc:
        raise RuntimeError("Il pacchetto 'python-docx' non e' installato. Installalo con: pip install python-docx") from exc

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)

    PRIMARY_HEX = "1B365D"
    STRIPE_HEX = "F4F7FA"
    BORDER_HEX = "CBD5E1"
    ACCENT_RGB = RGBColor(0x1B, 0x36, 0x5D)
    MUTED_RGB = RGBColor(0x47, 0x55, 0x69)
    BODY_RGB = RGBColor(0x1E, 0x29, 0x3B)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(9.0)
    normal_style.font.color.rgb = BODY_RGB

    def _set_cell_shading(cell, hex_color):
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    def _set_cell_margins(cell, top=65, bottom=65, left=85, right=85):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = OxmlElement("w:tcMar")
        for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
            node = OxmlElement(f"w:{m}")
            node.set(qn("w:w"), str(val))
            node.set(qn("w:type"), "dxa")
            tcMar.append(node)
        tcPr.append(tcMar)

    def _set_table_borders(table, border_color=BORDER_HEX):
        tblPr = table._tbl.tblPr
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>'
            f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="{PRIMARY_HEX}"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>'
            f'<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr.append(borders)

    def _prevent_row_split(row):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))

    def _repeat_header_row(row):
        row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))

    net_cfg = cfg.get("network", {}) if isinstance(cfg, dict) else {}
    net_name = str(net_cfg.get("name", getattr(net, "name", "Rete_Termica")))
    t_amb = float(net_cfg.get("ambient_temp_K", 300.0))

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run(f"Report Risultati Simulazione — {net_name}")
    hrun.font.size = Pt(8.5)
    hrun.font.color.rgb = MUTED_RGB

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = fp.add_run("Esportazione risultati rete termica criogenica e parametri rami")
    frun.font.size = Pt(8.5)
    frun.font.color.rgb = MUTED_RGB

    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    trun = title_p.add_run(f"Risultati Simulazione e Parametri dei Rami — {net_name}")
    trun.font.size = Pt(15.5)
    trun.font.bold = True
    trun.font.color.rgb = ACCENT_RGB

    ss = getattr(sol, "stage_summary", None) or {}
    q1_l = float(ss.get("Q1_load_W", np.nan))
    q1_c = float(ss.get("Q1_cap_W", np.nan))
    q2_l = float(ss.get("Q2_load_W", np.nan))
    q2_c = float(ss.get("Q2_cap_W", np.nan))
    m1 = (q1_c - q1_l) if (np.isfinite(q1_c) and np.isfinite(q1_l)) else np.nan
    m2 = (q2_c - q2_l) if (np.isfinite(q2_c) and np.isfinite(q2_l)) else np.nan
    max_res = float(ss.get("max_residual_W", np.nan))
    infeas = ss.get("infeasibility") or []

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(8)
    srun = sub_p.add_run(
        f"Converged: {getattr(sol, 'converged', False)}  |  "
        f"T_amb: {t_amb:.2f} K  |  "
        f"Max residuo: {max_res:.3e} W  |  "
        f"Stage 1 Margine: {m1:+.4f} W (Q1_load={q1_l:.4f} W, Q1_cap={q1_c:.4f} W)  |  "
        f"Stage 2 Margine: {m2:+.4f} W (Q2_load={q2_l:.4f} W, Q2_cap={q2_c:.4f} W)"
    )
    srun.font.size = Pt(8.8)
    srun.font.color.rgb = MUTED_RGB

    if infeas:
        warn_p = doc.add_paragraph()
        warn_p.paragraph_format.space_before = Pt(0)
        warn_p.paragraph_format.space_after = Pt(8)
        wrun = warn_p.add_run("ATTENZIONE - Infeasibilita fisica: " + "; ".join(infeas))
        wrun.font.size = Pt(9.0)
        wrun.font.bold = True
        wrun.font.color.rgb = RGBColor(0xB4, 0x53, 0x09)

    if isinstance(opt_info, dict) and opt_info:
        opt_p = doc.add_paragraph()
        opt_p.paragraph_format.space_before = Pt(0)
        opt_p.paragraph_format.space_after = Pt(8)
        orun = opt_p.add_run(
            f"Ottimizzazione attiva — Obiettivo: {opt_info.get('objective_mode', 'N/D')} | "
            f"Variabili ottimizzate: {opt_info.get('n_vars', 0)} | "
            f"Rami: {', '.join(opt_info.get('branch_ids') or sorted({s[0] for s in opt_info.get('slots', [])}))}"
        )
        orun.font.size = Pt(8.8)
        orun.font.bold = True
        orun.font.color.rgb = ACCENT_RGB

    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(4)
    h1.paragraph_format.space_after = Pt(4)
    h1r = h1.add_run("1. Temperature Nodali")
    h1r.font.size = Pt(11.0)
    h1r.font.bold = True
    h1r.font.color.rgb = ACCENT_RGB

    t_map = getattr(sol, "temperatures_K", None) or {}
    nodes_list = cfg.get("nodes", []) if isinstance(cfg, dict) else []
    node_keys = [str(n.get("id")) for n in nodes_list] if nodes_list else list(t_map.keys())
    if node_keys:
        ntbl = doc.add_table(rows=2, cols=len(node_keys))
        ntbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        _set_table_borders(ntbl)
        _prevent_row_split(ntbl.rows[0])
        _prevent_row_split(ntbl.rows[1])
        _repeat_header_row(ntbl.rows[0])
        for idx, nid in enumerate(node_keys):
            ch = ntbl.rows[0].cells[idx]
            cv = ntbl.rows[1].cells[idx]
            _set_cell_shading(ch, PRIMARY_HEX)
            _set_cell_margins(ch, top=55, bottom=55, left=70, right=70)
            ph = ch.paragraphs[0]
            ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rh = ph.add_run(nid)
            rh.font.size = Pt(8.5)
            rh.font.bold = True
            rh.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

            _set_cell_shading(cv, STRIPE_HEX)
            _set_cell_margins(cv, top=55, bottom=55, left=70, right=70)
            pv = cv.paragraphs[0]
            pv.alignment = WD_ALIGN_PARAGRAPH.CENTER
            val_t = float(t_map.get(nid, np.nan))
            rv = pv.add_run(f"{val_t:.3f} K" if np.isfinite(val_t) else "N/D")
            rv.font.size = Pt(8.8)
            rv.font.bold = True

    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)
    h2r = h2.add_run("2. Parametri dei Rami, Caduta di Temperatura (Delta T) e Potenza Termica (Q)")
    h2r.font.size = Pt(11.0)
    h2r.font.bold = True
    h2r.font.color.rgb = ACCENT_RGB

    headers = [
        "ID Ramo", "Tipo", "Da -> A", "Lunghezza\n[m]", "Sezione / Area\n[mq]",
        "Materiale / Parametri", "T_from / T_to\n[K]", "Delta T = T_from-T_to\n[K]", "Potenza Q\n[W]",
    ]
    col_widths = [
        Inches(1.25), Inches(0.80), Inches(1.25), Inches(0.75), Inches(0.85),
        Inches(2.45), Inches(1.05), Inches(1.00), Inches(1.00),
    ]

    br_cfg_map = {str(b.get("id", "")): b for b in (cfg.get("branches", []) if isinstance(cfg, dict) else [])}
    sol_br_list = getattr(sol, "branch_results", None) or []
    sol_br_map = {str(r.get("id", "")): r for r in sol_br_list}

    ordered_ids = [str(b.get("id", "")) for b in (cfg.get("branches", []) if isinstance(cfg, dict) else [])]
    for r in sol_br_list:
        rid = str(r.get("id", ""))
        if rid not in ordered_ids:
            ordered_ids.append(rid)

    btbl = doc.add_table(rows=1 + len(ordered_ids), cols=len(headers))
    btbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(btbl)
    _repeat_header_row(btbl.rows[0])
    _prevent_row_split(btbl.rows[0])

    for j, htxt in enumerate(headers):
        cell = btbl.rows[0].cells[j]
        cell.width = col_widths[j]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _set_cell_shading(cell, PRIMARY_HEX)
        _set_cell_margins(cell, top=70, bottom=70, left=65, right=65)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in (0, 2, 5) else WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(htxt)
        run.font.size = Pt(8.3)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, bid in enumerate(ordered_ids, start=1):
        row = btbl.rows[i]
        _prevent_row_split(row)
        bg = STRIPE_HEX if i % 2 == 0 else "FFFFFF"

        bcfg = br_cfg_map.get(bid, {})
        rsol = sol_br_map.get(bid, {})
        btype = str(bcfg.get("branch_type", rsol.get("branch_type", "N/D")))
        fn = str(bcfg.get("from_node", rsol.get("from_node", "")))
        tn = str(bcfg.get("to_node", rsol.get("to_node", "")))
        params = bcfg.get("params", {}) if isinstance(bcfg.get("params"), dict) else {}

        length_str = f"{float(params['length_m']):.4g}" if "length_m" in params and params["length_m"] is not None else "-"
        area_str = f"{float(params['area_m2']):.3e}" if "area_m2" in params and params["area_m2"] is not None else "-"

        if btype == "conduction":
            phys_str = f"Materiale: {params.get('material', 'N/D')}"
        elif btype == "contact":
            if "conductance_WK" in params and params["conductance_WK"] is not None:
                phys_str = f"Conduttanza: {float(params['conductance_WK']):.4g} W/K"
            elif "resistance_KW" in params and params["resistance_KW"] is not None:
                phys_str = f"Resistenza: {float(params['resistance_KW']):.4g} K/W"
            else:
                phys_str = "Contatto termico"
        elif btype == "radiation":
            phys_str = f"eps_eff = {params.get('emissivity_eff', 'N/D')}, F = {params.get('view_factor', 'N/D')}"
        elif btype == "mli":
            phys_str = (
                f"{params.get('mode', 'MLI')}, {params.get('layers', 'N/D')} strati, "
                f"{params.get('layer_density_per_cm', 'N/D')} strati/cm, P={params.get('residual_pressure_Pa', 'N/D')} Pa"
            )
        else:
            phys_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "-"

        t1 = float(t_map.get(fn, np.nan))
        t2 = float(t_map.get(tn, np.nan))
        dt = (t1 - t2) if (np.isfinite(t1) and np.isfinite(t2)) else np.nan
        q_w = float(rsol.get("Q_from_to_W", np.nan))

        vals = [
            bid, btype, f"{fn} -> {tn}", length_str, area_str, phys_str,
            f"{t1:.2f} / {t2:.2f}" if (np.isfinite(t1) and np.isfinite(t2)) else "N/D",
            f"{dt:+.3f}" if np.isfinite(dt) else "N/D",
            f"{q_w:+.4f}" if np.isfinite(q_w) else "N/D",
        ]

        for j, vtxt in enumerate(vals):
            cell = row.cells[j]
            cell.width = col_widths[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_shading(cell, bg)
            _set_cell_margins(cell, top=55, bottom=55, left=65, right=65)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            if j in (0, 2, 5):
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            elif j in (3, 4, 7, 8):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(vtxt)
            run.font.size = Pt(8.3)
            if j in (0, 8):
                run.font.bold = True

    doc.save(out_path)
    return out_path


def run_full_gui():
    out_dir = HERE.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = default_config_dict()
    ensure_cryocooler_config(cfg)

    root = tk.Tk()
    root.title("Editor Completo Rete Termica Criostato")
    root.geometry("1220x800")

    top_frm = ttk.LabelFrame(root, text="Rete & Risolutore", padding=8)
    top_frm.pack(fill=tk.X, padx=10, pady=(8, 4))

    name_var = tk.StringVar(value=cfg["network"]["name"])
    amb_var = tk.StringVar(value=str(cfg["network"]["ambient_temp_K"]))
    tmin_var = tk.StringVar(value=str(cfg["network"]["t_min_K"]))
    tmax_var = tk.StringVar(value=str(cfg["network"]["t_max_K"]))

    ttk.Label(top_frm, text="Nome:").grid(row=0, column=0, padx=4, sticky="w")
    ttk.Entry(top_frm, textvariable=name_var, width=28).grid(row=0, column=1, padx=4)
    ttk.Label(top_frm, text="T_amb [K]:").grid(row=0, column=2, padx=4, sticky="w")
    ttk.Combobox(top_frm, textvariable=amb_var, values=["300.0", "295.0", "77.0"], width=10).grid(row=0, column=3, padx=4)
    ttk.Label(top_frm, text="T_min [K]:").grid(row=0, column=4, padx=4, sticky="w")
    ttk.Combobox(top_frm, textvariable=tmin_var, values=["2.0", "3.5", "4.0", "10.0"], width=10).grid(row=0, column=5, padx=4)
    ttk.Label(top_frm, text="T_max [K]:").grid(row=0, column=6, padx=4, sticky="w")
    ttk.Combobox(top_frm, textvariable=tmax_var, values=["300.0", "320.0", "100.0"], width=10).grid(row=0, column=7, padx=4)

    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    # ---------------- Scheda NODI ----------------
    tab_nodes = ttk.Frame(nb, padding=8)
    nb.add(tab_nodes, text="Nodi")

    node_param_cols = ("id", "node_type", "material", "mass_kg", "target_temp_K", "fixed_temp_K")

    def _node_extended_label(nd: dict) -> str:
        nid = str(nd.get("id", "")).strip().lower()
        node_names = {
            "ambient": "Ambiente esterno",
            "stage1": "Primo stadio del criocooler",
            "anchor_s1": "Ancoraggio termico al primo stadio",
            "stage2": "Secondo stadio del criocooler",
            "shield": "Schermo termico",
            "cold_mass": "Massa fredda principale",
        }
        if nid in node_names:
            return node_names[nid]
        if "anchor" in nid or "anc" in nid:
            return "Ancoraggio termico"
        if "shield" in nid:
            return "Schermo termico"
        if "cold" in nid or "magnet" in nid or "cm" in nid:
            return "Massa fredda"
        if "stage" in nid:
            return "Stadio del criocooler"
        if "amb" in nid:
            return "Ambiente esterno"
        return str(nd.get("id", "")).strip()

    def _describe_node_full(nd: dict) -> str:
        nid = str(nd.get("id", "")).strip().lower()
        ntype = str(nd.get("node_type", "")).strip().lower()
        mat = str(nd.get("material", "")).strip()
        mass = nd.get("mass_kg", "")
        tgt = nd.get("target_temp_K", "")
        fix = nd.get("fixed_temp_K", None)

        id_map = {
            "ambient": "Sorgente termica esterna a temperatura ambiente",
            "stage1": "Stadio caldo/intermedio del criocooler con mappa di capacita frigorifera",
            "anchor_s1": "Ancoraggio termico collegato al primo stadio per intercettare il calore",
            "stage2": "Stadio freddo del criocooler con mappa di capacita frigorifera a bassa temperatura",
            "shield": "Schermo termico radiativo interposto tra ambiente e massa fredda",
            "cold_mass": "Massa fredda / sistema criogenico da mantenere alla temperatura operativa",
        }
        id_txt = id_map.get(nid, "Elemento o punto termico della rete del criostato")

        type_map = {
            "boundary": "condizione al contorno a temperatura imposta",
            "stage": "stadio del criocooler con potenza frigorifera dipendente da T",
            "thermal_mass": "massa termica dinamica con capacita termica propria",
        }
        type_txt = type_map.get(ntype, ntype or "N/D")

        mat_map = {
            "OFHC_Cu_RRR100": "Rame OFHC RRR 100 (alta conducibilita termica)",
            "Al6061_T6": "Alluminio 6061-T6",
            "G10_Normal": "Vetroresina G10 (isolante termico/strutturale)",
            "SS304": "Acciaio inox AISI 304",
            "Ti6Al4V": "Titanio Ti-6Al-4V",
            "Inconel718": "Inconel 718",
            "Inconel725": "Inconel 725",
            "Brass": "Ottone",
            "REBCO": "Nastro superconduttore REBCO",
        }
        mat_txt = mat_map.get(mat, mat if mat else "nessun materiale solido")

        try:
            m_val = float(mass)
            mass_txt = "senza massa inerziale (0 kg)" if m_val == 0.0 else f"massa {m_val:g} kg"
        except Exception:
            mass_txt = f"massa {mass}"

        fix_txt = "temperatura libera" if fix in ("", None) else f"temperatura fissa {fix} K"
        return f"{id_txt} | {type_txt} | {mat_txt} | {mass_txt} | T_rif={tgt} K | {fix_txt}"

    node_cols = node_param_cols + ("nome_per_esteso", "descrizione_aggiuntiva")
    node_tree = ttk.Treeview(tab_nodes, columns=node_cols, show="headings", height=8)
    for c in node_param_cols:
        node_tree.heading(c, text=c)
        node_tree.column(c, width=125, anchor="center")
    node_tree.heading("nome_per_esteso", text="Nome per esteso")
    node_tree.column("nome_per_esteso", width=240, anchor="w")
    node_tree.heading("descrizione_aggiuntiva", text="Descrizione aggiuntiva")
    node_tree.column("descrizione_aggiuntiva", width=520, anchor="w")
    node_tree.pack(fill=tk.BOTH, expand=True)

    node_edit_frm = ttk.Frame(tab_nodes, padding=(0, 6))
    node_edit_frm.pack(fill=tk.X)
    n_vars = {c: tk.StringVar() for c in node_cols}

    node_id_cb = ttk.Combobox(node_edit_frm, textvariable=n_vars["id"], width=15)
    node_type_cb = ttk.Combobox(node_edit_frm, textvariable=n_vars["node_type"], values=NODE_TYPE_OPTIONS, width=15)
    node_mat_cb = ttk.Combobox(node_edit_frm, textvariable=n_vars["material"], values=MATERIAL_OPTIONS, width=16)
    node_mass_cb = ttk.Combobox(node_edit_frm, textvariable=n_vars["mass_kg"], values=["0.0", "0.4", "1.5", "2.5", "8.0", "35.0"], width=12)
    node_tgt_cb = ttk.Combobox(node_edit_frm, textvariable=n_vars["target_temp_K"], values=["4.3", "4.5", "48.0", "50.0", "55.0", "300.0"], width=13)
    node_fix_cb = ttk.Combobox(node_edit_frm, textvariable=n_vars["fixed_temp_K"], values=FIXED_TEMP_OPTIONS, width=13)

    for idx, (lbl, wdg) in enumerate([
        ("id", node_id_cb),
        ("node_type", node_type_cb),
        ("material", node_mat_cb),
        ("mass_kg", node_mass_cb),
        ("target_temp_K", node_tgt_cb),
        ("fixed_temp_K", node_fix_cb),
    ]):
        ttk.Label(node_edit_frm, text=lbl).grid(row=0, column=idx, padx=3)
        wdg.grid(row=1, column=idx, padx=3)

    # ---------------- Scheda RAMI ----------------
    tab_branches = ttk.Frame(nb, padding=8)
    nb.add(tab_branches, text="Rami (Conduzione / Contatto / MLI / Radiazione)")

    br_param_cols = ("id", "branch_type", "from_node", "to_node", "params_json")

    def _branch_extended_label(br: dict) -> str:
        bid = str(br.get("id", "")).strip().lower()
        branch_names = {
            "cond_anchor_s1": "Conduzione tra ancoraggio e primo stadio",
            "cnt_anchor_s1": "Contatto termico tra ancoraggio e primo stadio",
            "link_s1_shield": "Collegamento conduttivo tra primo stadio e schermo termico",
            "cnt_s1_shield": "Contatto termico tra primo stadio e schermo termico",
            "mli_amb_shield": "Isolamento multistrato (MLI) tra ambiente e schermo termico",
            "rad_amb_shield": "Irraggiamento tra ambiente e schermo termico",
            "rad_shield_cm": "Irraggiamento tra schermo termico e massa fredda",
            "busbar_s2_cm": "Sbarra conduttiva tra secondo stadio e massa fredda",
            "cnt_busbar_cm": "Contatto termico tra secondo stadio e massa fredda",
            "lead_amb_anc": "Cavo / alimentazione tra ambiente e ancoraggio",
            "lead_anc_cm": "Cavo / alimentazione tra ancoraggio e massa fredda",
            "strut_amb_anc": "Puntone strutturale tra ambiente e ancoraggio",
            "strut_anc_cm": "Puntone strutturale tra ancoraggio e massa fredda",
            "diag_anc_cm": "Tirante diagonale tra ancoraggio e massa fredda",
        }
        if bid in branch_names:
            return branch_names[bid]
        if "strut" in bid:
            return "Puntone strutturale"
        if "diag" in bid:
            return "Tirante diagonale"
        if "lead" in bid:
            return "Cavo / alimentazione elettrica"
        if "busbar" in bid:
            return "Sbarra conduttiva"
        if "mli" in bid:
            return "Isolamento multistrato (MLI)"
        if "rad" in bid:
            return "Scambio termico radiativo"
        if "cnt" in bid:
            return "Contatto termico"
        return str(br.get("id", "")).strip()

    def _describe_branch_full(br: dict) -> str:
        btype = str(br.get("branch_type", "")).strip().lower()
        fn = str(br.get("from_node", "")).strip()
        tn = str(br.get("to_node", "")).strip()
        p = br.get("params", {}) if isinstance(br.get("params"), dict) else {}
        type_map = {
            "conduction": "Conduzione termica in solido",
            "contact": "Resistenza/conduttanza di contatto all'interfaccia",
            "mli": "Isolamento termico multistrato (Multi-Layer Insulation)",
            "radiation": "Scambio termico per irraggiamento",
        }
        t_desc = type_map.get(btype, btype or "N/D")
        if btype == "conduction":
            p_desc = (
                f"materiale {p.get('material', 'N/D')}, "
                f"area {p.get('area_m2', 'N/D')} m2, lunghezza {p.get('length_m', 'N/D')} m"
            )
        elif btype == "contact":
            if "resistance_KW" in p and "conductance_WK" not in p:
                p_desc = f"resistenza di contatto {p.get('resistance_KW', 'N/D')} K/W"
            else:
                p_desc = f"conduttanza di contatto {p.get('conductance_WK', 'N/D')} W/K"
        elif btype == "mli":
            p_desc = (
                f"modalita {p.get('mode', 'N/D')}, {p.get('layers', 'N/D')} strati, "
                f"densita {p.get('layer_density_per_cm', 'N/D')} strati/cm, "
                f"area {p.get('area_m2', 'N/D')} m2, pressione {p.get('residual_pressure_Pa', 'N/D')} Pa"
            )
        elif btype == "radiation":
            p_desc = (
                f"area {p.get('area_m2', 'N/D')} m2, emissivita {p.get('emissivity_eff', 'N/D')}, "
                f"fattore di vista {p.get('view_factor', 'N/D')}"
            )
        else:
            p_desc = "parametri del ramo"
        return f"{t_desc} da {fn} a {tn} | {p_desc}"

    br_cols = br_param_cols + ("nome_per_esteso", "descrizione_aggiuntiva")
    br_tree = ttk.Treeview(tab_branches, columns=br_cols, show="headings", height=9)
    br_widths = {"id": 150, "branch_type": 130, "from_node": 125, "to_node": 125, "params_json": 340}
    for c in br_param_cols:
        br_tree.heading(c, text="params" if c == "params_json" else c)
        br_tree.column(c, width=br_widths[c], anchor="w" if c == "params_json" else "center")
    br_tree.heading("nome_per_esteso", text="Nome per esteso")
    br_tree.column("nome_per_esteso", width=260, anchor="w")
    br_tree.heading("descrizione_aggiuntiva", text="Descrizione aggiuntiva")
    br_tree.column("descrizione_aggiuntiva", width=520, anchor="w")
    br_tree.pack(fill=tk.BOTH, expand=True)

    br_edit_frm = ttk.Frame(tab_branches, padding=(0, 6))
    br_edit_frm.pack(fill=tk.X)
    b_vars = {
        "id": tk.StringVar(),
        "branch_type": tk.StringVar(value="conduction"),
        "from_node": tk.StringVar(),
        "to_node": tk.StringVar(),
        "cond_material": tk.StringVar(value="OFHC_Cu_RRR100"),
        "cond_area": tk.StringVar(value="5e-4"),
        "cond_length": tk.StringVar(value="0.04"),
        "cnt_mode": tk.StringVar(value="conductance_WK"),
        "cnt_value": tk.StringVar(value="25.0"),
        "rad_area": tk.StringVar(value="0.45"),
        "rad_emissivity": tk.StringVar(value="0.02"),
        "rad_view_factor": tk.StringVar(value="1.0"),
        "mli_mode": tk.StringVar(value="parametric_modified_lockheed"),
        "mli_area": tk.StringVar(value="0.65"),
        "mli_layers": tk.StringVar(value="40"),
        "mli_density": tk.StringVar(value="25.0"),
        "mli_pressure": tk.StringVar(value="1e-5"),
        "params_preview": tk.StringVar(value="{}"),
    }

    br_id_cb = ttk.Combobox(br_edit_frm, textvariable=b_vars["id"], width=16)
    br_type_cb = ttk.Combobox(br_edit_frm, textvariable=b_vars["branch_type"], values=BRANCH_TYPE_OPTIONS, width=13)
    br_from_cb = ttk.Combobox(br_edit_frm, textvariable=b_vars["from_node"], width=12)
    br_to_cb = ttk.Combobox(br_edit_frm, textvariable=b_vars["to_node"], width=12)

    ttk.Label(br_edit_frm, text="id").grid(row=0, column=0)
    br_id_cb.grid(row=1, column=0, padx=2)
    ttk.Label(br_edit_frm, text="branch_type").grid(row=0, column=1)
    br_type_cb.grid(row=1, column=1, padx=2)
    ttk.Label(br_edit_frm, text="from_node").grid(row=0, column=2)
    br_from_cb.grid(row=1, column=2, padx=2)
    ttk.Label(br_edit_frm, text="to_node").grid(row=0, column=3)
    br_to_cb.grid(row=1, column=3, padx=2)

    br_guided_frm = ttk.LabelFrame(tab_branches, text="Parametri Ramo Guidati (Menu a tendina)", padding=(6, 4))
    br_guided_frm.pack(fill=tk.X, pady=(2, 4))

    frm_cond = ttk.Frame(br_guided_frm)
    cond_mat_choices = [m for m in MATERIAL_OPTIONS if m]
    ttk.Label(frm_cond, text="material").grid(row=0, column=0, padx=3)
    cond_mat_cb = ttk.Combobox(frm_cond, textvariable=b_vars["cond_material"], values=cond_mat_choices, width=16)
    cond_mat_cb.grid(row=1, column=0, padx=3)
    ttk.Label(frm_cond, text="area_m2").grid(row=0, column=1, padx=3)
    ttk.Combobox(frm_cond, textvariable=b_vars["cond_area"], values=AREA_PRESET_OPTIONS, width=12).grid(row=1, column=1, padx=3)
    ttk.Label(frm_cond, text="length_m").grid(row=0, column=2, padx=3)
    ttk.Combobox(frm_cond, textvariable=b_vars["cond_length"], values=LENGTH_PRESET_OPTIONS, width=12).grid(row=1, column=2, padx=3)

    frm_cnt = ttk.Frame(br_guided_frm)
    ttk.Label(frm_cnt, text="parametro contatto").grid(row=0, column=0, padx=3)
    ttk.Combobox(frm_cnt, textvariable=b_vars["cnt_mode"], values=CONTACT_PARAM_MODE_OPTIONS, width=16).grid(row=1, column=0, padx=3)
    ttk.Label(frm_cnt, text="valore").grid(row=0, column=1, padx=3)
    ttk.Combobox(frm_cnt, textvariable=b_vars["cnt_value"], values=CONTACT_VAL_PRESET_OPTIONS, width=12).grid(row=1, column=1, padx=3)

    frm_rad = ttk.Frame(br_guided_frm)
    ttk.Label(frm_rad, text="area_m2").grid(row=0, column=0, padx=3)
    ttk.Combobox(frm_rad, textvariable=b_vars["rad_area"], values=AREA_PRESET_OPTIONS, width=12).grid(row=1, column=0, padx=3)
    ttk.Label(frm_rad, text="emissivity_eff").grid(row=0, column=1, padx=3)
    ttk.Combobox(frm_rad, textvariable=b_vars["rad_emissivity"], values=EMISSIVITY_PRESET_OPTIONS, width=12).grid(row=1, column=1, padx=3)
    ttk.Label(frm_rad, text="view_factor").grid(row=0, column=2, padx=3)
    ttk.Combobox(frm_rad, textvariable=b_vars["rad_view_factor"], values=VIEW_FACTOR_PRESET_OPTIONS, width=12).grid(row=1, column=2, padx=3)

    frm_mli = ttk.Frame(br_guided_frm)
    ttk.Label(frm_mli, text="mode").grid(row=0, column=0, padx=3)
    ttk.Combobox(frm_mli, textvariable=b_vars["mli_mode"], values=MLI_MODE_OPTIONS, width=28).grid(row=1, column=0, padx=3)
    ttk.Label(frm_mli, text="area_m2").grid(row=0, column=1, padx=3)
    ttk.Combobox(frm_mli, textvariable=b_vars["mli_area"], values=AREA_PRESET_OPTIONS, width=11).grid(row=1, column=1, padx=3)
    ttk.Label(frm_mli, text="layers").grid(row=0, column=2, padx=3)
    ttk.Combobox(frm_mli, textvariable=b_vars["mli_layers"], values=MLI_LAYERS_PRESET_OPTIONS, width=9).grid(row=1, column=2, padx=3)
    ttk.Label(frm_mli, text="layer_density_per_cm").grid(row=0, column=3, padx=3)
    ttk.Combobox(frm_mli, textvariable=b_vars["mli_density"], values=MLI_DENSITY_PRESET_OPTIONS, width=11).grid(row=1, column=3, padx=3)
    ttk.Label(frm_mli, text="residual_pressure_Pa").grid(row=0, column=4, padx=3)
    ttk.Combobox(frm_mli, textvariable=b_vars["mli_pressure"], values=MLI_PRESSURE_PRESET_OPTIONS, width=12).grid(row=1, column=4, padx=3)

    _type_frames = {"conduction": frm_cond, "contact": frm_cnt, "radiation": frm_rad, "mli": frm_mli}

    def _populate_guided_from_params(btype: str, p_dict: dict):
        p = p_dict if isinstance(p_dict, dict) else {}
        if btype == "conduction":
            b_vars["cond_material"].set(str(p.get("material", "OFHC_Cu_RRR100")))
            b_vars["cond_area"].set(str(p.get("area_m2", 5e-4)))
            b_vars["cond_length"].set(str(p.get("length_m", 0.04)))
        elif btype == "contact":
            if "resistance_KW" in p and "conductance_WK" not in p:
                b_vars["cnt_mode"].set("resistance_KW")
                b_vars["cnt_value"].set(str(p.get("resistance_KW", 0.04)))
            else:
                b_vars["cnt_mode"].set("conductance_WK")
                b_vars["cnt_value"].set(str(p.get("conductance_WK", 25.0)))
        elif btype == "radiation":
            b_vars["rad_area"].set(str(p.get("area_m2", 0.45)))
            b_vars["rad_emissivity"].set(str(p.get("emissivity_eff", 0.02)))
            b_vars["rad_view_factor"].set(str(p.get("view_factor", 1.0)))
        elif btype == "mli":
            b_vars["mli_mode"].set(str(p.get("mode", "parametric_modified_lockheed")))
            b_vars["mli_area"].set(str(p.get("area_m2", 0.65)))
            b_vars["mli_layers"].set(str(p.get("layers", 40)))
            b_vars["mli_density"].set(str(p.get("layer_density_per_cm", 25.0)))
            b_vars["mli_pressure"].set(str(p.get("residual_pressure_Pa", 1e-5)))

    def _collect_guided_params(fallback_params: dict | None = None) -> dict:
        return build_branch_params_from_guided(
            branch_type=b_vars["branch_type"].get(),
            cond_material=b_vars["cond_material"].get(),
            cond_area=b_vars["cond_area"].get(),
            cond_length=b_vars["cond_length"].get(),
            cnt_mode=b_vars["cnt_mode"].get(),
            cnt_value=b_vars["cnt_value"].get(),
            rad_area=b_vars["rad_area"].get(),
            rad_emissivity=b_vars["rad_emissivity"].get(),
            rad_view_factor=b_vars["rad_view_factor"].get(),
            mli_mode=b_vars["mli_mode"].get(),
            mli_area=b_vars["mli_area"].get(),
            mli_layers=b_vars["mli_layers"].get(),
            mli_density=b_vars["mli_density"].get(),
            mli_pressure=b_vars["mli_pressure"].get(),
            fallback_params=fallback_params,
        )

    def _refresh_params_preview(*_evt):
        try:
            p_cur = _collect_guided_params()
            b_vars["params_preview"].set(json.dumps(p_cur))
        except Exception as exc:
            b_vars["params_preview"].set(f"<valore non valido: {exc}>")

    def _on_branch_type_change(_evt=None):
        btype = b_vars["branch_type"].get().strip() or "conduction"
        for k, f_w in _type_frames.items():
            if k == btype:
                f_w.grid(row=0, column=0, sticky="w")
            else:
                f_w.grid_remove()
        _refresh_params_preview()

    for _vk in (
        "cond_material", "cond_area", "cond_length",
        "cnt_mode", "cnt_value",
        "rad_area", "rad_emissivity", "rad_view_factor",
        "mli_mode", "mli_area", "mli_layers", "mli_density", "mli_pressure",
    ):
        b_vars[_vk].trace_add("write", _refresh_params_preview)

    cond_mat_cb.bind("<<ComboboxSelected>>", _refresh_params_preview)
    cond_mat_cb.bind("<FocusOut>", _refresh_params_preview)
    br_type_cb.bind("<<ComboboxSelected>>", _on_branch_type_change)
    br_type_cb.bind("<FocusOut>", _on_branch_type_change)
    ttk.Label(br_guided_frm, text="params sincronizzati:").grid(row=1, column=0, sticky="w", padx=3, pady=(4, 0))
    ttk.Label(br_guided_frm, textvariable=b_vars["params_preview"], font=("Consolas", 8)).grid(row=1, column=1, columnspan=4, sticky="w", padx=3, pady=(4, 0))
    _on_branch_type_change()

    # ---------------- Scheda CARICHI ----------------
    tab_loads = ttk.Frame(nb, padding=8)
    nb.add(tab_loads, text="Carichi Termici (HeatLoad)")

    ld_param_cols = ("id", "node", "power_W", "category", "enabled")

    def _load_extended_label(ld: dict) -> str:
        lid = str(ld.get("id", "")).strip().lower()
        load_names = {
            "joule_lead_amb_anc": "Riscaldamento Joule cavi tra ambiente e ancoraggio",
            "joule_lead_anc_cm": "Riscaldamento Joule cavi tra ancoraggio e massa fredda",
            "ac_losses_cm": "Perdite AC sulla massa fredda",
        }
        if lid in load_names:
            return load_names[lid]
        if "joule" in lid or "lead" in lid:
            return "Riscaldamento Joule sui cavi"
        if "ac_loss" in lid or "ac_" in lid:
            return "Perdite AC"
        if "eddy" in lid:
            return "Correnti parassite"
        return str(ld.get("id", "")).strip()

    def _describe_load_full(ld: dict) -> str:
        node = str(ld.get("node", "")).strip()
        pwr = ld.get("power_W", "")
        cat = str(ld.get("category", "")).strip()
        en = bool(ld.get("enabled", False))
        cat_map = {
            "joule_lead": "Dissipazione per effetto Joule nei conduttori",
            "ac_loss": "Perdite in corrente alternata (AC losses)",
            "eddy_current": "Correnti parassite indotte",
            "radiation_leak": "Infiltrazione termica radiativa",
            "static_load": "Carico termico statico",
            "custom": "Carico termico personalizzato",
        }
        cat_txt = cat_map.get(cat, cat or "N/D")
        st_txt = "attivo nel bilancio termico" if en else "disabilitato (escluso dal bilancio)"
        return f"{cat_txt} | potenza {pwr} W applicata al nodo {node} | {st_txt}"

    ld_cols = ld_param_cols + ("nome_per_esteso", "descrizione_aggiuntiva")
    ld_tree = ttk.Treeview(tab_loads, columns=ld_cols, show="headings", height=8)
    for c in ld_param_cols:
        ld_tree.heading(c, text=c)
        ld_tree.column(c, width=130, anchor="center")
    ld_tree.heading("nome_per_esteso", text="Nome per esteso")
    ld_tree.column("nome_per_esteso", width=260, anchor="w")
    ld_tree.heading("descrizione_aggiuntiva", text="Descrizione aggiuntiva")
    ld_tree.column("descrizione_aggiuntiva", width=520, anchor="w")
    ld_tree.pack(fill=tk.BOTH, expand=True)

    ld_edit_frm = ttk.Frame(tab_loads, padding=(0, 6))
    ld_edit_frm.pack(fill=tk.X)
    l_vars = {c: tk.StringVar() for c in ld_cols}

    ld_id_cb = ttk.Combobox(ld_edit_frm, textvariable=l_vars["id"], width=20)
    ld_node_cb = ttk.Combobox(ld_edit_frm, textvariable=l_vars["node"], width=15)
    ld_pwr_cb = ttk.Combobox(ld_edit_frm, textvariable=l_vars["power_W"], values=["0.0", "0.01", "0.05", "0.15", "0.20", "0.50"], width=12)
    ld_cat_cb = ttk.Combobox(ld_edit_frm, textvariable=l_vars["category"], values=LOAD_CATEGORY_OPTIONS, width=16)
    ld_en_cb = ttk.Combobox(ld_edit_frm, textvariable=l_vars["enabled"], values=BOOL_OPTIONS, width=10)

    for idx, (lbl, wdg) in enumerate([
        ("id", ld_id_cb),
        ("node", ld_node_cb),
        ("power_W", ld_pwr_cb),
        ("category", ld_cat_cb),
        ("enabled", ld_en_cb),
    ]):
        ttk.Label(ld_edit_frm, text=lbl).grid(row=0, column=idx, padx=3)
        wdg.grid(row=1, column=idx, padx=3)

    # ---------------- Scheda CRYOCOOLER ----------------
    tab_cc = ttk.Frame(nb, padding=8)
    nb.add(tab_cc, text="Cryocooler (Stage 1 & 2)")

    cc_top_frm = ttk.Frame(tab_cc)
    cc_top_frm.pack(fill=tk.X, pady=(0, 6))
    cc_model_var = tk.StringVar(value=cfg["cryocooler"]["model"])
    cc_scale_var = tk.StringVar(value=str(cfg["cryocooler"]["cooling_power_scale"]))

    ttk.Label(cc_top_frm, text="Modello:").grid(row=0, column=0, padx=4, sticky="w")
    ttk.Combobox(cc_top_frm, textvariable=cc_model_var, values=["two_stage_map", "gm_two_stage", "pt_two_stage"], width=18).grid(row=0, column=1, padx=4)
    ttk.Label(cc_top_frm, text="Fattore scala Q_cap:").grid(row=0, column=2, padx=4, sticky="w")
    ttk.Combobox(cc_top_frm, textvariable=cc_scale_var, values=["0.75", "0.90", "1.0", "1.10", "1.25"], width=10).grid(row=0, column=3, padx=4)

    cc_stages_frm = ttk.Frame(tab_cc)
    cc_stages_frm.pack(fill=tk.BOTH, expand=True)

    stage_ui = {}
    for col_idx, st_key in enumerate(("stage1", "stage2")):
        st_cfg = cfg["cryocooler"][st_key]
        box = ttk.LabelFrame(cc_stages_frm, text=f"{st_key.upper()} — Mappa Capacita Q_cap(T)", padding=6)
        box.grid(row=0, column=col_idx, padx=6, sticky="nsew")
        cc_stages_frm.columnconfigure(col_idx, weight=1)

        s_vars = {
            "enabled": tk.StringVar(value=str(bool(st_cfg["enabled"]))),
            "node_id": tk.StringVar(value=str(st_cfg["node_id"])),
            "t_ref_K": tk.StringVar(value=str(st_cfg["t_ref_K"])),
            "q_ref_W": tk.StringVar(value=str(st_cfg["q_ref_W"])),
            "pt_T": tk.StringVar(value=str(st_cfg["t_grid_K"][0])),
            "pt_Q": tk.StringVar(value=str(st_cfg["q_cap_W"][0])),
            "eval_T": tk.StringVar(value=str(st_cfg["t_ref_K"])),
            "eval_res": tk.StringVar(value=""),
        }

        meta_frm = ttk.Frame(box)
        meta_frm.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(meta_frm, text="Attivo:").grid(row=0, column=0, padx=2)
        ttk.Combobox(meta_frm, textvariable=s_vars["enabled"], values=BOOL_OPTIONS, width=7).grid(row=0, column=1, padx=2)
        ttk.Label(meta_frm, text="Nodo:").grid(row=0, column=2, padx=2)
        ttk.Combobox(meta_frm, textvariable=s_vars["node_id"], values=["stage1", "stage2", "shield", "cold_mass"], width=10).grid(row=0, column=3, padx=2)
        ttk.Label(meta_frm, text="T_ref [K]:").grid(row=0, column=4, padx=2)
        ttk.Entry(meta_frm, textvariable=s_vars["t_ref_K"], width=8).grid(row=0, column=5, padx=2)
        ttk.Label(meta_frm, text="Q_ref [W]:").grid(row=0, column=6, padx=2)
        ttk.Entry(meta_frm, textvariable=s_vars["q_ref_W"], width=8).grid(row=0, column=7, padx=2)

        map_tree = ttk.Treeview(box, columns=("T_K", "Q_cap_W"), show="headings", height=6)
        map_tree.heading("T_K", text="T [K]")
        map_tree.column("T_K", width=110, anchor="center")
        map_tree.heading("Q_cap_W", text="Q_cap [W]")
        map_tree.column("Q_cap_W", width=110, anchor="center")
        map_tree.pack(fill=tk.BOTH, expand=True)

        pt_frm = ttk.Frame(box)
        pt_frm.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(pt_frm, text="T [K]:").grid(row=0, column=0, padx=2)
        ttk.Entry(pt_frm, textvariable=s_vars["pt_T"], width=8).grid(row=0, column=1, padx=2)
        ttk.Label(pt_frm, text="Q_cap [W]:").grid(row=0, column=2, padx=2)
        ttk.Entry(pt_frm, textvariable=s_vars["pt_Q"], width=8).grid(row=0, column=3, padx=2)

        eval_frm = ttk.Frame(box)
        eval_frm.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(eval_frm, text="Test T [K]:").grid(row=0, column=0, padx=2)
        ttk.Entry(eval_frm, textvariable=s_vars["eval_T"], width=8).grid(row=0, column=1, padx=2)
        ttk.Label(eval_frm, textvariable=s_vars["eval_res"], width=24).grid(row=0, column=3, padx=4, sticky="w")

        stage_ui[st_key] = {"vars": s_vars, "tree": map_tree, "pt_frm": pt_frm, "eval_frm": eval_frm}

    def refresh_cryocooler_ui():
        ensure_cryocooler_config(cfg)
        cc_model_var.set(str(cfg["cryocooler"]["model"]))
        cc_scale_var.set(str(cfg["cryocooler"]["cooling_power_scale"]))
        scale_val = float(cfg["cryocooler"].get("cooling_power_scale", 1.0))
        for st_key, ui in stage_ui.items():
            st_cfg = cfg["cryocooler"][st_key]
            sv = ui["vars"]
            sv["enabled"].set(str(bool(st_cfg["enabled"])))
            sv["node_id"].set(str(st_cfg["node_id"]))
            sv["t_ref_K"].set(str(st_cfg["t_ref_K"]))
            sv["q_ref_W"].set(str(st_cfg["q_ref_W"]))
            tree = ui["tree"]
            tree.delete(*tree.get_children())
            for t_v, q_v in zip(st_cfg["t_grid_K"], st_cfg["q_cap_W"]):
                tree.insert("", tk.END, values=[f"{float(t_v):.3f}", f"{float(q_v):.4f}"])
            try:
                t_test = float(sv["eval_T"].get())
            except Exception:
                t_test = float(st_cfg["t_ref_K"])
                sv["eval_T"].set(str(t_test))
            q_interp = interpolate_stage_capacity(st_cfg, t_test, scale=scale_val)
            sv["eval_res"].set(f"Q_cap({t_test:.2f} K) = {q_interp:.4f} W")

    def apply_cryocooler_meta():
        ensure_cryocooler_config(cfg)
        cfg["cryocooler"]["model"] = cc_model_var.get().strip() or "two_stage_map"
        cfg["cryocooler"]["cooling_power_scale"] = float(cc_scale_var.get())
        for st_key, ui in stage_ui.items():
            sv = ui["vars"]
            st_cfg = cfg["cryocooler"][st_key]
            en_str = sv["enabled"].get().strip().lower()
            st_cfg["enabled"] = en_str in ("true", "1", "yes", "si", "si")
            st_cfg["node_id"] = sv["node_id"].get().strip() or st_key
            st_cfg["t_ref_K"] = float(sv["t_ref_K"].get())
            st_cfg["q_ref_W"] = float(sv["q_ref_W"].get())
        refresh_cryocooler_ui()

    def make_map_callbacks(st_key):
        ui = stage_ui[st_key]
        tree = ui["tree"]
        sv = ui["vars"]

        def on_sel(_evt=None):
            sel = tree.selection()
            if not sel:
                return
            vals = tree.item(sel[0], "values")
            sv["pt_T"].set(vals[0])
            sv["pt_Q"].set(vals[1])

        def upd_pt():
            sel = tree.selection()
            if not sel:
                return
            idx = tree.index(sel[0])
            st_cfg = cfg["cryocooler"][st_key]
            st_cfg["t_grid_K"][idx] = float(sv["pt_T"].get())
            st_cfg["q_cap_W"][idx] = float(sv["pt_Q"].get())
            pairs = sorted(zip(st_cfg["t_grid_K"], st_cfg["q_cap_W"]), key=lambda p: p[0])
            st_cfg["t_grid_K"] = [p[0] for p in pairs]
            st_cfg["q_cap_W"] = [p[1] for p in pairs]
            refresh_cryocooler_ui()

        def add_pt():
            st_cfg = cfg["cryocooler"][st_key]
            st_cfg["t_grid_K"].append(float(sv["pt_T"].get()))
            st_cfg["q_cap_W"].append(float(sv["pt_Q"].get()))
            pairs = sorted(zip(st_cfg["t_grid_K"], st_cfg["q_cap_W"]), key=lambda p: p[0])
            st_cfg["t_grid_K"] = [p[0] for p in pairs]
            st_cfg["q_cap_W"] = [p[1] for p in pairs]
            refresh_cryocooler_ui()

        def del_pt():
            sel = tree.selection()
            if not sel:
                return
            st_cfg = cfg["cryocooler"][st_key]
            if len(st_cfg["t_grid_K"]) <= 1:
                messagebox.showwarning("Attenzione", "Deve restare almeno un punto nella mappa.")
                return
            idx = tree.index(sel[0])
            st_cfg["t_grid_K"].pop(idx)
            st_cfg["q_cap_W"].pop(idx)
            refresh_cryocooler_ui()

        def eval_pt():
            apply_cryocooler_meta()

        tree.bind("<<TreeviewSelect>>", on_sel)
        ttk.Button(ui["pt_frm"], text="Aggiorna Punto", command=upd_pt).grid(row=0, column=4, padx=2)
        ttk.Button(ui["pt_frm"], text="Aggiungi", command=add_pt).grid(row=0, column=5, padx=2)
        ttk.Button(ui["pt_frm"], text="Elimina", command=del_pt).grid(row=0, column=6, padx=2)
        ttk.Button(ui["eval_frm"], text="Valuta Q_cap(T)", command=eval_pt).grid(row=0, column=2, padx=4)

    make_map_callbacks("stage1")
    make_map_callbacks("stage2")

    cc_act_frm = ttk.Frame(tab_cc)
    cc_act_frm.pack(fill=tk.X, pady=(6, 0))
    ttk.Button(cc_act_frm, text="Applica Parametri & Mappe Cryocooler", command=apply_cryocooler_meta).pack(side=tk.LEFT, padx=4)
    refresh_cryocooler_ui()

    def sync_combobox_choices():
        node_ids = [nd["id"] for nd in cfg["nodes"]]
        branch_ids = [br["id"] for br in cfg["branches"]]
        load_ids = [ld["id"] for ld in cfg["loads"]]
        node_id_cb["values"] = node_ids
        br_from_cb["values"] = node_ids
        br_to_cb["values"] = node_ids
        ld_node_cb["values"] = node_ids
        br_id_cb["values"] = branch_ids
        ld_id_cb["values"] = load_ids

    def refresh_node_tree():
        node_tree.delete(*node_tree.get_children())
        for nd in cfg["nodes"]:
            base_vals = [
                nd["id"], nd["node_type"], nd.get("material", ""),
                nd.get("mass_kg", ""), nd.get("target_temp_K", ""),
                "" if nd.get("fixed_temp_K") is None else nd.get("fixed_temp_K"),
            ]
            row_vals = list(base_vals) + [_node_extended_label(nd), _describe_node_full(nd)]
            node_tree.insert("", tk.END, values=row_vals)
        sync_combobox_choices()

    def on_select_node(_evt=None):
        sel = node_tree.selection()
        if not sel:
            return
        vals = node_tree.item(sel[0], "values")
        col_to_val = dict(zip(node_cols, vals))
        for c in node_param_cols:
            n_vars[c].set(col_to_val.get(c, ""))

    def update_node():
        sel = node_tree.selection()
        if not sel:
            return
        idx = node_tree.index(sel[0])
        cfg["nodes"][idx] = {
            "id": n_vars["id"].get().strip(),
            "node_type": n_vars["node_type"].get().strip(),
            "material": n_vars["material"].get().strip(),
            "mass_kg": float(n_vars["mass_kg"].get()) if n_vars["mass_kg"].get().strip() else 0.0,
            "target_temp_K": float(n_vars["target_temp_K"].get()) if n_vars["target_temp_K"].get().strip() else 0.0,
            "fixed_temp_K": float(n_vars["fixed_temp_K"].get()) if n_vars["fixed_temp_K"].get().strip() else None,
        }
        refresh_node_tree()

    def add_node():
        nid = n_vars["id"].get().strip()
        if not nid:
            messagebox.showwarning("Attenzione", "Inserisci o seleziona un id nodo.")
            return
        cfg["nodes"].append({
            "id": nid,
            "node_type": n_vars["node_type"].get().strip() or "thermal_mass",
            "material": n_vars["material"].get().strip(),
            "mass_kg": float(n_vars["mass_kg"].get()) if n_vars["mass_kg"].get().strip() else 1.0,
            "target_temp_K": float(n_vars["target_temp_K"].get()) if n_vars["target_temp_K"].get().strip() else 10.0,
            "fixed_temp_K": float(n_vars["fixed_temp_K"].get()) if n_vars["fixed_temp_K"].get().strip() else None,
        })
        refresh_node_tree()

    def del_node():
        sel = node_tree.selection()
        if not sel:
            return
        cfg["nodes"].pop(node_tree.index(sel[0]))
        refresh_node_tree()

    node_tree.bind("<<TreeviewSelect>>", on_select_node)
    ttk.Button(node_edit_frm, text="Aggiorna Nodo", command=update_node).grid(row=1, column=6, padx=4)
    ttk.Button(node_edit_frm, text="Aggiungi", command=add_node).grid(row=1, column=7, padx=2)
    ttk.Button(node_edit_frm, text="Elimina", command=del_node).grid(row=1, column=8, padx=2)

    def refresh_branch_tree(select_index: int | None = None):
        br_tree.delete(*br_tree.get_children())
        item_ids = []
        for br in cfg["branches"]:
            base_vals = [br["id"], br["branch_type"], br["from_node"], br["to_node"], json.dumps(br["params"])]
            row_vals = list(base_vals) + [_branch_extended_label(br), _describe_branch_full(br)]
            iid = br_tree.insert("", tk.END, values=row_vals)
            item_ids.append(iid)
        sync_combobox_choices()
        if select_index is not None and 0 <= select_index < len(item_ids):
            target_iid = item_ids[select_index]
            br_tree.selection_set(target_iid)
            br_tree.focus(target_iid)
            br_tree.see(target_iid)

    def on_select_branch(_evt=None):
        sel = br_tree.selection()
        if not sel:
            return
        idx = br_tree.index(sel[0])
        vals = br_tree.item(sel[0], "values")
        col_to_val = dict(zip(br_cols, vals))
        b_vars["id"].set(col_to_val.get("id", ""))
        btype = str(col_to_val.get("branch_type", "conduction")).strip() or "conduction"
        b_vars["branch_type"].set(btype)
        b_vars["from_node"].set(col_to_val.get("from_node", ""))
        b_vars["to_node"].set(col_to_val.get("to_node", ""))
        p_dict = cfg["branches"][idx].get("params", {}) if 0 <= idx < len(cfg["branches"]) else {}
        raw_p = col_to_val.get("params_json", "")
        if not p_dict and raw_p:
            try:
                p_dict = json.loads(raw_p)
            except Exception:
                p_dict = {}
        _populate_guided_from_params(btype, p_dict)
        _on_branch_type_change()

    def update_branch():
        sel = br_tree.selection()
        if not sel:
            return
        idx = br_tree.index(sel[0])
        existing_params = cfg["branches"][idx].get("params", {}) if 0 <= idx < len(cfg["branches"]) else {}
        try:
            p_obj = _collect_guided_params(fallback_params=existing_params)
        except Exception as exc:
            messagebox.showerror("Parametro non valido", f"Errore nei parametri del ramo: {exc}")
            return
        cfg["branches"][idx] = {
            "id": b_vars["id"].get().strip(),
            "branch_type": b_vars["branch_type"].get().strip(),
            "from_node": b_vars["from_node"].get().strip(),
            "to_node": b_vars["to_node"].get().strip(),
            "params": p_obj,
        }
        refresh_branch_tree(select_index=idx)

    def add_branch():
        bid = b_vars["id"].get().strip()
        if not bid:
            messagebox.showwarning("Attenzione", "Inserisci o seleziona un id ramo.")
            return
        try:
            p_obj = _collect_guided_params(fallback_params=None)
        except Exception as exc:
            messagebox.showerror("Parametro non valido", f"Errore nei parametri del ramo: {exc}")
            return
        cfg["branches"].append({
            "id": bid,
            "branch_type": b_vars["branch_type"].get().strip() or "conduction",
            "from_node": b_vars["from_node"].get().strip() or "ambient",
            "to_node": b_vars["to_node"].get().strip() or "shield",
            "params": p_obj,
        })
        refresh_branch_tree(select_index=len(cfg["branches"]) - 1)

    def del_branch():
        sel = br_tree.selection()
        if not sel:
            return
        cfg["branches"].pop(br_tree.index(sel[0]))
        refresh_branch_tree()

    br_tree.bind("<<TreeviewSelect>>", on_select_branch)
    ttk.Button(br_edit_frm, text="Aggiorna Ramo", command=update_branch).grid(row=1, column=5, padx=3)
    ttk.Button(br_edit_frm, text="Aggiungi", command=add_branch).grid(row=1, column=6, padx=2)
    ttk.Button(br_edit_frm, text="Elimina", command=del_branch).grid(row=1, column=7, padx=2)

    def refresh_load_tree():
        ld_tree.delete(*ld_tree.get_children())
        for ld in cfg["loads"]:
            base_vals = [ld["id"], ld["node"], ld["power_W"], ld["category"], str(bool(ld["enabled"]))]
            row_vals = list(base_vals) + [_load_extended_label(ld), _describe_load_full(ld)]
            ld_tree.insert("", tk.END, values=row_vals)
        sync_combobox_choices()

    def on_select_load(_evt=None):
        sel = ld_tree.selection()
        if not sel:
            return
        vals = ld_tree.item(sel[0], "values")
        col_to_val = dict(zip(ld_cols, vals))
        for c in ld_param_cols:
            l_vars[c].set(col_to_val.get(c, ""))

    def update_load():
        sel = ld_tree.selection()
        if not sel:
            return
        idx = ld_tree.index(sel[0])
        en_str = l_vars["enabled"].get().strip().lower()
        cfg["loads"][idx] = {
            "id": l_vars["id"].get().strip(),
            "node": l_vars["node"].get().strip(),
            "power_W": float(l_vars["power_W"].get()),
            "category": l_vars["category"].get().strip(),
            "enabled": en_str in ("true", "1", "yes", "si", "si"),
        }
        refresh_load_tree()

    def add_load():
        lid = l_vars["id"].get().strip()
        if not lid:
            messagebox.showwarning("Attenzione", "Inserisci o seleziona un id carico.")
            return
        en_str = l_vars["enabled"].get().strip().lower()
        cfg["loads"].append({
            "id": lid,
            "node": l_vars["node"].get().strip() or "cold_mass",
            "power_W": float(l_vars["power_W"].get() or 0.0),
            "category": l_vars["category"].get().strip() or "custom",
            "enabled": en_str in ("true", "1", "yes", "si", "si"),
        })
        refresh_load_tree()

    def del_load():
        sel = ld_tree.selection()
        if not sel:
            return
        cfg["loads"].pop(ld_tree.index(sel[0]))
        refresh_load_tree()

    ld_tree.bind("<<TreeviewSelect>>", on_select_load)
    ttk.Button(ld_edit_frm, text="Aggiorna Carico", command=update_load).grid(row=1, column=5, padx=4)
    ttk.Button(ld_edit_frm, text="Aggiungi", command=add_load).grid(row=1, column=6, padx=2)
    ttk.Button(ld_edit_frm, text="Elimina", command=del_load).grid(row=1, column=7, padx=2)

    refresh_node_tree()
    refresh_branch_tree()
    refresh_load_tree()

    # ---------------- Scheda JSON ----------------
    tab_json = ttk.Frame(nb, padding=8)
    nb.add(tab_json, text="Configurazione Completa (JSON)")
    json_txt = tk.Text(tab_json, font=("Consolas", 9))
    json_txt.pack(fill=tk.BOTH, expand=True)

    def sync_to_json_tab():
        apply_cryocooler_meta()
        json_txt.delete("1.0", tk.END)
        json_txt.insert(tk.END, json.dumps(cfg, indent=2))

    def apply_from_json_tab():
        nonlocal cfg
        try:
            new_cfg = json.loads(json_txt.get("1.0", tk.END))
            cfg = new_cfg
            ensure_cryocooler_config(cfg)
            name_var.set(str(cfg["network"]["name"]))
            amb_var.set(str(cfg["network"]["ambient_temp_K"]))
            tmin_var.set(str(cfg["network"].get("t_min_K", 3.5)))
            tmax_var.set(str(cfg["network"].get("t_max_K", 300.0)))
            refresh_node_tree()
            refresh_branch_tree()
            refresh_load_tree()
            refresh_cryocooler_ui()
            messagebox.showinfo("JSON applicato", "Configurazione aggiornata dalle tabelle.")
        except Exception as exc:
            messagebox.showerror("Errore JSON", str(exc))

    j_btn_frm = ttk.Frame(tab_json, padding=(0, 4))
    j_btn_frm.pack(fill=tk.X)
    ttk.Button(j_btn_frm, text="Aggiorna JSON dalle tabelle", command=sync_to_json_tab).pack(side=tk.LEFT, padx=4)
    ttk.Button(j_btn_frm, text="Applica JSON alle tabelle", command=apply_from_json_tab).pack(side=tk.LEFT, padx=4)
    sync_to_json_tab()

    # ---------------- Riquadro Azioni & Risultati ----------------
    bot_frm = ttk.LabelFrame(root, text="Azioni & Risultati", padding=8)
    bot_frm.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 8))

    out_txt = tk.Text(bot_frm, height=9, font=("Consolas", 9))
    out_txt.pack(fill=tk.X, pady=(0, 6))

    state = {"net": None, "sol": None, "fig": None, "win": None, "last_opt_info": None}

    def sync_top_fields():
        cfg["network"]["name"] = name_var.get().strip() or "Rete_Termica"
        cfg["network"]["ambient_temp_K"] = float(amb_var.get())
        cfg["network"]["t_min_K"] = float(tmin_var.get())
        cfg["network"]["t_max_K"] = float(tmax_var.get())
        apply_cryocooler_meta()

    def _format_solution_lines(sol, opt_info=None):
        ss = sol.stage_summary or {}
        q1_l = ss.get("Q1_load_W", np.nan)
        q1_c = ss.get("Q1_cap_W", np.nan)
        q1_cm = ss.get("Q1_cap_max_W", q1_c)
        q2_l = ss.get("Q2_load_W", np.nan)
        q2_c = ss.get("Q2_cap_W", np.nan)
        q2_cm = ss.get("Q2_cap_max_W", q2_c)
        m1_max = (q1_cm - q1_l) if np.isfinite(q1_cm) and np.isfinite(q1_l) else np.nan
        m2_max = (q2_cm - q2_l) if np.isfinite(q2_cm) and np.isfinite(q2_l) else np.nan
        max_res = ss.get("max_residual_W", np.nan)
        res_str = f"{max_res:.3e} W" if np.isfinite(max_res) else "N/A"
        infeas = ss.get("infeasibility") or []
        head = (
            f"[OTTIMIZZAZIONE OPZIONALE] Converged: {sol.converged} | Variabili: {opt_info.get('n_vars', 0)} | Obiettivo: {opt_info.get('objective_mode', 'N/D')}"
            if opt_info else
            f"Converged: {sol.converged} | Max residuo nodale: {res_str} | {sol.message}"
        )
        lines = [
            head,
            f"Stage 1: T={sol.temperatures_K.get('stage1', np.nan):.2f} K | Q1_load={q1_l:.4f} W | Q1_cap={q1_c:.4f} W | Margine_max(Qcap_max-load)={m1_max:+.4f} W",
            f"Stage 2: T={sol.temperatures_K.get('stage2', np.nan):.3f} K | Q2_load={q2_l:.4f} W | Q2_cap={q2_c:.4f} W | Margine_max(Qcap_max-load)={m2_max:+.4f} W",
        ]
        if infeas:
            lines.append("ATTENZIONE Infeasibilita fisica: " + "; ".join(infeas))
        for w in (sol.warnings or []):
            lines.append(f"[WARNING] {w}")
        lines.append("Nodi: " + ", ".join(f"{k}={v:.2f}K" for k, v in sol.temperatures_K.items()))
        lines.append("Rami: " + ", ".join(f"{r['id']}={r['Q_from_to_W']:+.4f}W" for r in sol.branch_results))
        return lines

    def solve_now():
        import traceback
        if "opt_enabled_var" in locals() and opt_enabled_var.get():
            run_optimization_now(True)
            return
        try:
            sync_top_fields()
            net = build_network_from_config(cfg)
            sol = run_solver_with_cryocooler(net, cfg)
            state["net"] = net
            state["sol"] = sol
            state["last_opt_info"] = None
            lines = _format_solution_lines(sol, None)
            out_txt.delete("1.0", tk.END)
            out_txt.insert(tk.END, "\n".join(lines))
        except Exception as exc:
            tb_text = traceback.format_exc()
            print(tb_text)
            out_txt.delete("1.0", tk.END)
            out_txt.insert(tk.END, tb_text)
            messagebox.showerror("Errore simulazione", f"{exc}\n\nDettaglio stampato in console e nel riquadro risultati.")

    def open_graph():
        if state["net"] is None or state["sol"] is None:
            solve_now()
        if state["net"] is None or state["sol"] is None:
            return
        if state["fig"] is not None:
            plt.close(state["fig"])
        if state["win"] is not None and state["win"].winfo_exists():
            state["win"].destroy()
        fig = make_topology_figure(state["net"], state["sol"])
        state["fig"] = fig
        win = tk.Toplevel(root)
        win.title("Grafo Topologia Criostato")
        state["win"] = win
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ------------------------------------------------------------------
    # Save / Load JSON in formato canonico (con fallback legacy)
    # ------------------------------------------------------------------
    def _load_project_io_module():
        """Import lazy di io/project_io con fallback su importlib."""
        try:
            from io import project_io as _pio  # type: ignore[import-not-found]
            return _pio
        except Exception:
            import importlib.util as _ilu
            pio_path = HERE / "io" / "project_io.py"
            if not pio_path.exists():
                return None
            spec = _ilu.spec_from_file_location("project_io", str(pio_path))
            if spec is None or spec.loader is None:
                return None
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    def save_json_file():
        sync_top_fields()
        p = filedialog.asksaveasfilename(
            initialdir=str(out_dir),
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not p:
            return
        pio = _load_project_io_module()
        try:
            if pio is not None and hasattr(pio, "cfg_to_canonical"):
                payload = pio.cfg_to_canonical(cfg)
            else:
                # Fallback: salva in formato legacy GUI
                payload = cfg
            Path(p).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            messagebox.showinfo("Salvato", f"Configurazione salvata in:\n{p}")
        except Exception as exc:
            messagebox.showerror("Errore salvataggio JSON", str(exc))

    def load_json_file():
        nonlocal cfg
        p = filedialog.askopenfilename(
            initialdir=str(out_dir), filetypes=[("JSON", "*.json")]
        )
        if not p:
            return
        pio = _load_project_io_module()
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            if pio is not None and hasattr(pio, "canonical_to_cfg"):
                cfg = pio.canonical_to_cfg(data)
            else:
                # Fallback: assume formato legacy GUI
                if "network" not in data:
                    raise ValueError(
                        "Formato JSON non riconosciuto e 'io/project_io.py' non disponibile. "
                        "Atteso formato legacy (con 'network') o canonico (con 'project')."
                    )
                cfg = data
            ensure_cryocooler_config(cfg)
            name_var.set(str(cfg["network"]["name"]))
            amb_var.set(str(cfg["network"]["ambient_temp_K"]))
            tmin_var.set(str(cfg["network"].get("t_min_K", 3.5)))
            tmax_var.set(str(cfg["network"].get("t_max_K", 300.0)))
            refresh_node_tree()
            refresh_branch_tree()
            refresh_load_tree()
            refresh_cryocooler_ui()
            sync_to_json_tab()
            solve_now()
            messagebox.showinfo("Caricato", f"Configurazione caricata da:\n{p}")
        except Exception as exc:
            messagebox.showerror("Errore caricamento JSON", str(exc))

    def _is_solution_valid(sol):
        if sol is None:
            return False, "Nessun risultato: esegui prima il calcolo."
        if not sol.converged:
            return False, "Risultato NON convergiuto: export marcato INVALID."
        ss = sol.stage_summary or {}
        if ss.get("infeasibility"):
            return False, "Infeasibilita fisica: export marcato INVALID."
        return True, ""

    def export_png_csv():
        if state["net"] is None or state["sol"] is None:
            solve_now()
        if state["net"] is None or state["sol"] is None:
            return
        valid, msg = _is_solution_valid(state["sol"])
        prefix = "" if valid else "INVALID_"
        fig = state["fig"] if state["fig"] is not None else make_topology_figure(state["net"], state["sol"])
        png_p = out_dir / f"{prefix}topologia_criostato_full.png"
        fig.savefig(png_p, dpi=300, bbox_inches="tight")
        if state["fig"] is None:
            plt.close(fig)
        csv_p = out_dir / f"{prefix}risultati_full.csv"
        ss = state["sol"].stage_summary or {}
        with open(csv_p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Meta", "Converged", state["sol"].converged])
            w.writerow(["Meta", "Message", state["sol"].message])
            w.writerow(["Meta", "Max_residual_W", ss.get("max_residual_W", float("nan"))])
            w.writerow(["Meta", "Q1_load_W", ss.get("Q1_load_W", float("nan"))])
            w.writerow(["Meta", "Q1_cap_W", ss.get("Q1_cap_W", float("nan"))])
            w.writerow(["Meta", "Q2_load_W", ss.get("Q2_load_W", float("nan"))])
            w.writerow(["Meta", "Q2_cap_W", ss.get("Q2_cap_W", float("nan"))])
            for w_msg in (state["sol"].warnings or []):
                w.writerow(["Warning", w_msg])
            for inf in (ss.get("infeasibility") or []):
                w.writerow(["Infeasibility", inf])
            w.writerow(["Tipo", "Id", "Grandezza", "Valore"])
            for k, v in state["sol"].temperatures_K.items():
                w.writerow(["Nodo", k, "T_K", f"{v:.6f}"])
            for r in state["sol"].branch_results:
                w.writerow(["Ramo", r["id"], "Q_from_to_W", f"{r['Q_from_to_W']:+.6f}"])
        suffix = "" if valid else f"\n\nATTENZIONE: {msg}\nFile marcati con prefisso INVALID_."
        messagebox.showinfo("Esportato", f"PNG: {png_p}\nCSV: {csv_p}{suffix}")

    def export_results_doc():
        if state["net"] is None or state["sol"] is None:
            solve_now()
        if state["net"] is None or state["sol"] is None:
            return
        sync_top_fields()
        valid, msg = _is_solution_valid(state["sol"])
        prefix = "" if valid else "INVALID_"
        default_doc = out_dir / f"{prefix}risultati_rete_rami.docx"
        p = filedialog.asksaveasfilename(
            initialdir=str(out_dir),
            initialfile=default_doc.name,
            defaultextension=".docx",
            filetypes=[("Documento Word (.docx)", "*.docx")],
        )
        if not p:
            return
        try:
            saved_p = export_results_report_doc(
                cfg=cfg,
                net=state["net"],
                sol=state["sol"],
                out_path=Path(p),
                opt_info=state.get("last_opt_info"),
            )
            suffix = "" if valid else f"\n\nATTENZIONE: {msg}"
            messagebox.showinfo("Documento Esportato", f"Documento risultati salvato in:\n{saved_p}{suffix}")
        except Exception as exc:
            import traceback
            tb_text = traceback.format_exc()
            print(tb_text)
            out_txt.delete("1.0", tk.END)
            out_txt.insert(tk.END, tb_text)
            messagebox.showerror("Errore esportazione documento", str(exc))

    # ---------------- Ottimizzazione ----------------
    opt_frm = ttk.LabelFrame(bot_frm, text="Dimensionamento & Ottimizzazione (Opzionale: Sezioni / Lunghezze)", padding=6)
    opt_frm.pack(fill=tk.X, pady=(0, 6))

    opt_enabled_var = tk.BooleanVar(value=False)
    opt_area_var = tk.BooleanVar(value=True)
    opt_length_var = tk.BooleanVar(value=True)
    opt_smin_var = tk.StringVar(value="0.5")
    opt_smax_var = tk.StringVar(value="2.0")
    opt_iter_var = tk.StringVar(value="20")
    opt_q1_max_var = tk.StringVar(value="0.0")
    opt_q2_max_var = tk.StringVar(value="0.0")

    sizable_ids = get_sizable_conduction_branches(cfg)
    default_opt_branches = [
        bid for bid in ("cond_anchor_s1", "link_s1_shield", "busbar_s2_cm", "strut_amb_anc", "strut_anc_cm", "lead_amb_anc", "lead_anc_cm")
        if bid in sizable_ids
    ] or sizable_ids

    ttk.Checkbutton(opt_frm, text="Abilita su 1. Risolvi Rete", variable=opt_enabled_var).grid(row=0, column=0, padx=4, sticky="w")
    ttk.Checkbutton(opt_frm, text="Ottimizza area_m2", variable=opt_area_var).grid(row=0, column=1, padx=4, sticky="w")
    ttk.Checkbutton(opt_frm, text="Ottimizza length_m", variable=opt_length_var).grid(row=0, column=2, padx=4, sticky="w")

    ttk.Label(opt_frm, text="Obiettivo:").grid(row=0, column=3, padx=(8, 2), sticky="e")
    obj_labels_list = [OPT_OBJECTIVE_LABELS[k] for k in OPT_OBJECTIVE_OPTIONS]
    obj_label_to_key = {OPT_OBJECTIVE_LABELS[k]: k for k in OPT_OBJECTIVE_OPTIONS}
    opt_obj_combo_var = tk.StringVar(value=OPT_OBJECTIVE_LABELS["combined_margin_grad"])
    ttk.Combobox(opt_frm, textvariable=opt_obj_combo_var, values=obj_labels_list, width=36, state="readonly").grid(row=0, column=4, padx=2)

    ttk.Label(opt_frm, text="Scala [min,max]:").grid(row=0, column=5, padx=(6, 2), sticky="e")
    ttk.Entry(opt_frm, textvariable=opt_smin_var, width=6).grid(row=0, column=6, padx=1)
    ttk.Entry(opt_frm, textvariable=opt_smax_var, width=6).grid(row=0, column=7, padx=1)
    ttk.Label(opt_frm, text="Iter:").grid(row=0, column=8, padx=(4, 2), sticky="e")
    ttk.Entry(opt_frm, textvariable=opt_iter_var, width=5).grid(row=0, column=9, padx=1)

    ttk.Label(opt_frm, text="Rami:").grid(row=1, column=0, padx=4, pady=(4, 0), sticky="w")
    opt_branches_var = tk.StringVar(value=", ".join(default_opt_branches))
    opt_branches_cb = ttk.Combobox(
        opt_frm,
        textvariable=opt_branches_var,
        values=[", ".join(default_opt_branches), ", ".join(sizable_ids)] + sizable_ids,
        width=55,
    )
    opt_branches_cb.grid(row=1, column=1, columnspan=5, padx=2, pady=(4, 0), sticky="we")

    def run_optimization_now(apply_to_cfg: bool = True):
        nonlocal cfg
        import traceback
        try:
            sync_top_fields()
            raw_b = [x.strip() for x in opt_branches_var.get().split(",") if x.strip()]
            obj_key = obj_label_to_key.get(opt_obj_combo_var.get(), "combined_margin_grad")
            s_min = float(opt_smin_var.get())
            s_max = float(opt_smax_var.get())
            n_it = int(float(opt_iter_var.get()))
            Q1_max = float(opt_q1_max_var.get().strip() or "0")
            Q2_max = float(opt_q2_max_var.get().strip() or "0")

            best_cfg, net, sol, info = run_optional_sizing_optimization(
                base_cfg=cfg,
                branch_ids=raw_b,
                opt_area=bool(opt_area_var.get()),
                opt_length=bool(opt_length_var.get()),
                scale_min=s_min,
                scale_max=s_max,
                objective_mode=obj_key,
                max_iter=n_it,
                Q1_load_max_W=Q1_max,
                Q2_load_max_W=Q2_max,
            )
            if apply_to_cfg:
                cfg = best_cfg
                refresh_node_tree()
                refresh_branch_tree()
                refresh_load_tree()
                refresh_cryocooler_ui()
                sync_to_json_tab()

            state["net"] = net
            state["sol"] = sol
            state["last_opt_info"] = {**info, "objective_mode": obj_key, "branch_ids": raw_b}
            lines = _format_solution_lines(sol, state["last_opt_info"])
            out_txt.delete("1.0", tk.END)
            out_txt.insert(tk.END, "\n".join(lines))
        except Exception as exc:
            tb_text = traceback.format_exc()
            print(tb_text)
            out_txt.delete("1.0", tk.END)
            out_txt.insert(tk.END, tb_text)
            messagebox.showerror("Errore ottimizzazione", f"{exc}\n\nDettaglio stampato in console e nel riquadro risultati.")

    ttk.Button(opt_frm, text="Ottimizza Sezioni/Lunghezze", command=lambda: run_optimization_now(True)).grid(row=1, column=6, columnspan=4, padx=4, pady=(4, 0), sticky="e")
    ttk.Label(opt_frm, text="Q1_max [W] (0=ignora):").grid(row=2, column=0, padx=(4, 2), pady=(4, 0), sticky="w")
    ttk.Entry(opt_frm, textvariable=opt_q1_max_var, width=8).grid(row=2, column=1, padx=1, pady=(4, 0))
    ttk.Label(opt_frm, text="Q2_max [W] (0=ignora):").grid(row=2, column=2, padx=(8, 2), pady=(4, 0), sticky="e")
    ttk.Entry(opt_frm, textvariable=opt_q2_max_var, width=8).grid(row=2, column=3, padx=1, pady=(4, 0))

    # ================================================================

    # Apri GUI Transient (Toplevel in-process, nessun file JSON)

    # ================================================================

    def _open_transient_window():

        """Apre la GUI transient in una Toplevel, passandole la rete in memoria."""

        try:

            from gui_transient import launch_transient_window

        except ImportError as exc:

            messagebox.showerror(

                "gui_transient.py non trovato",

                f"File 'gui_transient.py' non trovato.\n\n{exc}"

            )

            return

        net = state.get("net")

        if net is None:

            try:

                sync_top_fields()

                net = build_network_from_config(cfg)

            except Exception as exc:

                messagebox.showerror(

                    "Impossibile costruire la rete",

                    f"Errore: {exc}\n\nRisolvi prima la rete o correggi i parametri."

                )

                return

        launch_transient_window(parent=root, net=net, cfg=cfg)


    act_frm = ttk.Frame(bot_frm)
    act_frm.pack(fill=tk.X)
    ttk.Button(act_frm, text="1. Risolvi Rete", command=solve_now).pack(side=tk.LEFT, padx=3)
    ttk.Button(act_frm, text="2. Apri / Aggiorna Grafo", command=open_graph).pack(side=tk.LEFT, padx=3)
    ttk.Button(act_frm, text="3. Esporta PNG + CSV", command=export_png_csv).pack(side=tk.LEFT, padx=3)
    ttk.Button(act_frm, text="4. Esporta Documento (.docx)", command=export_results_doc).pack(side=tk.LEFT, padx=3)

    ttk.Button(act_frm, text="5. Apri GUI Transient", command=_open_transient_window).pack(side=tk.LEFT, padx=3)
    ttk.Button(act_frm, text="Salva Config JSON", command=save_json_file).pack(side=tk.LEFT, padx=12)
    ttk.Button(act_frm, text="Carica Config JSON", command=load_json_file).pack(side=tk.LEFT, padx=3)

    solve_now()
    root.mainloop()


if __name__ == "__main__":
    run_full_gui()
