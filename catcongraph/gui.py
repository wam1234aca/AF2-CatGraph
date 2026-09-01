"""Local, bilingual browser interface for AF2-CatGraph.

The GUI writes ordinary AF2-CatGraph YAML files and starts the existing CLI. It
does not contain a second implementation of the scientific workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import base64
import os
import queue
import subprocess
import sys
import threading
from typing import Any, Callable

import yaml

from catcongraph.gui_config import (
    CONTACT_UNIT_CONSERVATIVE, CONTACT_UNIT_GUIDED, CONTACT_UNIT_UNIVERSAL,
    DEFAULT_VALUES, as_yaml_list,
    build_config, config_to_values, default_config_path, global_rename_config_path,
    global_rename_values,
    split_paths, write_config,
)
from catcongraph.gui_results import (
    class_interactions, class_rows, pymol_script, representative_pdb, result_files, viewer_html,
)
from catcongraph.gui_runtime import (
    build_identity, gui_status, launch_streamlit_gui, refresh_running_build,
)
from catcongraph.config import load_yaml_config, normalize_project_config, public_output_subdir
from catcongraph.profiles import validated_contact_unit_target
from catcongraph.runtime_env import environment_with_runtime_libraries
from catcongraph import __version__


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STEP_KEYS = [
    "preflight", "step01", "step02", "step03", "step04", "step05", "step06",
    "step07", "step08", "step09", "step10", "all",
]

PROJECT_WIDGET_KEYS = (
    "ccg_field_target_id", "ccg_field_output_root", "ccg_field_af2_dir",
    "ccg_field_template", "ccg_field_ligands", "ccg_field_conservation",
    "ccg_field_ged", "ccg_field_resnames", "ccg_field_manifest", "ccg_global_rename_id",
)


def _command_runtime_prefix(command: list[str]) -> Path:
    """Resolve the environment whose C++ runtime a background task must use."""
    if command:
        executable = Path(command[0]).expanduser()
        if executable.is_absolute() and executable.name.lower().startswith("python"):
            return executable.parent.parent
        if len(command) > 3 and command[1] == "run":
            environment_name = ""
            for option in ("-n", "--name"):
                if option in command and command.index(option) + 1 < len(command):
                    environment_name = command[command.index(option) + 1]
                    break
            if environment_name:
                probe = subprocess.run(
                    [command[0], "run", "-n", environment_name, "python", "-c",
                     "import sys; print(sys.prefix)"],
                    capture_output=True, text=True, timeout=30, check=False,
                    env=environment_with_runtime_libraries(prefix=sys.prefix),
                )
                if probe.returncode == 0 and probe.stdout.strip():
                    return Path(probe.stdout.strip().splitlines()[-1])
    return Path(sys.prefix)


@dataclass
class BackgroundJob:
    """A CLI process that survives normal Streamlit reruns."""

    command: list[str]
    cwd: Path
    process: subprocess.Popen[str] = field(init=False)
    lines: list[str] = field(default_factory=list)
    _queue: queue.Queue[str] = field(default_factory=queue.Queue, init=False)

    def __post_init__(self) -> None:
        env = environment_with_runtime_libraries(
            prefix=_command_runtime_prefix(self.command)
        )
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MPLBACKEND", "Agg")
        self.process = subprocess.Popen(
            self.command, cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert self.process.stdout is not None
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for line in iter(self.process.stdout.readline, ""):
            self._queue.put(line.rstrip("\n"))
        self.process.stdout.close()

    def refresh(self) -> None:
        while True:
            try:
                self.lines.append(self._queue.get_nowait())
            except queue.Empty:
                return

    @property
    def returncode(self) -> int | None:
        self.refresh()
        return self.process.poll()

    @property
    def running(self) -> bool:
        return self.returncode is None

    def stop(self) -> None:
        if self.running:
            self.process.terminate()


def launch_gui(
    address: str = "127.0.0.1", port: int = 8501, *, replace: bool = False,
    startup_config: str | Path | None = None, open_browser: bool = True,
) -> None:
    """Launch one identifiable Streamlit instance, keeping the CLI dependency-free."""
    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("The graphical interface needs Streamlit. Run: pip install -e '.[gui]'") from exc
    raise SystemExit(
        launch_streamlit_gui(
            REPOSITORY_ROOT, Path(__file__), __version__, address, port,
            replace=replace, startup_config=startup_config, open_browser=open_browser,
        )
    )


def runtime_status(address: str = "127.0.0.1", port: int = 8501) -> dict[str, Any]:
    """Expose the source hash and active launcher record for `gui --check`."""

    return gui_status(REPOSITORY_ROOT, __version__, address, port)


def _st():
    import streamlit as st
    return st


def _initialize_state(st: Any) -> None:
    server_token = os.environ.get("CATCONGRAPH_GUI_STATE_TOKEN", "interactive-development")
    if st.session_state.get("ccg_server_token") != server_token:
        for key in list(st.session_state):
            if str(key).startswith("ccg_"):
                del st.session_state[key]
        values = dict(DEFAULT_VALUES)
        config_path = str(default_config_path(values, REPOSITORY_ROOT))
        startup_config = os.environ.get("CATCONGRAPH_GUI_CONFIG", "").strip()
        startup_error = ""
        if startup_config:
            try:
                values = config_to_values(startup_config)
                config_path = startup_config
            except Exception as exc:
                startup_error = str(exc)
        st.session_state.ccg_server_token = server_token
        st.session_state.ccg_values = values
        st.session_state.ccg_config_path = config_path
        st.session_state.ccg_startup_error = startup_error
    st.session_state.setdefault("ccg_values", dict(DEFAULT_VALUES))
    st.session_state.setdefault("ccg_config_path", str(default_config_path(DEFAULT_VALUES, REPOSITORY_ROOT)))
    st.session_state.setdefault("ccg_job", None)
    st.session_state.setdefault("ccg_language", "English")
    st.session_state.setdefault("ccg_selected_class", "")
    st.session_state.setdefault("ccg_af2_preprocess_output", "")
    st.session_state.setdefault("ccg_equivalence_preview", [])
    st.session_state.setdefault("ccg_equivalence_preview_inventory", [])
    st.session_state.setdefault("ccg_equivalence_preview_report", {})
    st.session_state.setdefault("ccg_equivalence_preview_input_audit", [])
    st.session_state.setdefault("ccg_project_widget_generation", 0)


def _clear_project_widget_state(st: Any) -> None:
    # Streamlit can restore a form widget's browser-side value after a rerun,
    # even when its old session-state entry was removed. Advancing the widget
    # generation gives renamed/loaded projects a fresh set of widget identities
    # so the current page immediately renders the synchronized values.
    for state_key in list(st.session_state):
        if any(
            str(state_key) == base or str(state_key).startswith(f"{base}__")
            for base in PROJECT_WIDGET_KEYS
        ):
            st.session_state.pop(state_key, None)
    st.session_state.ccg_project_widget_generation = int(
        st.session_state.get("ccg_project_widget_generation", 0)
    ) + 1


def _project_widget_key(st: Any, base: str) -> str:
    generation = int(st.session_state.get("ccg_project_widget_generation", 0))
    return f"{base}__{generation}"


def _t(st: Any, zh: str, en: str) -> str:
    return en if st.session_state.get("ccg_language") == "English" else zh


def _values(st: Any) -> dict[str, Any]:
    return st.session_state.ccg_values


def _set_values(st: Any, updates: dict[str, Any]) -> None:
    merged = dict(_values(st))
    merged.update(updates)
    st.session_state.ccg_values = merged


def _step_labels(st: Any) -> dict[str, str]:
    zh = {
        "preflight": "00 · 运行前检查", "step01": "01 · AF2 / ColabFold 模型准备",
        "step02": "02 · 模板引导的配体放置", "step03": "03 · 空间碰撞筛选",
        "step04": "04 · 催化区域 pLDDT 质控", "step05": "05 · PLIP 相互作用图",
        "step06": "06 · 四条件关键残基筛选", "step07": "07 · GED 拓扑分类",
        "step08": "08 · 配体接触单元分组", "step09": "09 · 类别图和汇总表",
        "step10": "10 · 可选配体原子选择 / 裁剪",
        "all": "01–09 · 主流程",
    }
    en = {
        "preflight": "00 · Preflight check", "step01": "01 · AF2 / ColabFold model preparation",
        "step02": "02 · Template-guided ligand placement", "step03": "03 · Steric-clash filter",
        "step04": "04 · Catalytic-region pLDDT QC", "step05": "05 · PLIP interaction graphs",
        "step06": "06 · Four-criterion key-residue screening", "step07": "07 · GED topology classification",
        "step08": "08 · Ligand contact-unit grouping", "step09": "09 · Class plots and summary tables",
        "step10": "10 · Optional ligand-atom selection / trimming",
        "all": "01–09 · Main workflow",
    }
    return en if st.session_state.get("ccg_language") == "English" else zh


def _show_css(st: Any) -> None:
    """Apply the Studio's presentation layer without changing workflow state."""
    st.markdown("""
<style>
  :root {--ccg-ink:#173331;--ccg-muted:#58716e;--ccg-line:#d8e6e2;--ccg-soft:#f6faf8;
    --ccg-primary:#08736d;--ccg-primary-dark:#063b40;--ccg-accent:#a6cf62;}
  html, body, [class*="css"] {color:var(--ccg-ink);}
  .block-container {max-width:1320px;padding-top:4.15rem!important;padding-bottom:2.6rem;overflow:visible!important;}
  .block-container h1,.block-container h2,.block-container h3 {
    line-height:1.34!important;padding:.18em 0 .08em!important;margin-top:0!important;overflow:visible!important;
  }
  [data-testid="stHeadingWithActionElements"],
  [data-testid="stHeadingWithActionElements"] > div {overflow:visible!important;}
  [data-testid="stSidebar"] {background:linear-gradient(180deg,#063b40 0%,#075158 58%,#0a6065 100%);}
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
  [data-testid="stSidebar"] details summary,
  [data-testid="stSidebar"] [role="radiogroup"] label p {color:#f4fbf9!important;}
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {opacity:1;}
  [data-testid="stSidebar"] hr {border-color:rgba(231,248,244,.28);}
  [data-testid="stSidebar"] div[data-testid="stCodeBlock"] code {color:#183b39!important;}
  [data-testid="stSidebar"] div[data-testid="stCodeBlock"] {background:#f4fbf9;border-radius:10px;}
  [data-testid="stSidebar"] button {background:#f7fcfa!important;color:#0a4c4f!important;
    border:1px solid #c9e3dc!important;font-weight:650!important;}
  [data-testid="stSidebar"] button:hover {background:#e7f5f1!important;border-color:#a8d2c8!important;}
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] div[data-baseweb="select"] > div {background:#fff!important;color:#173331!important;}
  [data-testid="stSidebar"] div[data-baseweb="select"] span {color:#173331!important;}
  .ccg-hero {padding:1.55rem 1.7rem;border-radius:20px;color:#fff;box-shadow:0 12px 30px #073f4620;
    background:linear-gradient(120deg,var(--ccg-primary-dark),var(--ccg-primary) 58%,#82b74b);margin-bottom:1.15rem;}
  .ccg-hero h1 {margin:0;color:#fff;font-size:clamp(1.8rem,4vw,2.35rem);letter-spacing:-.025em;}
  .ccg-hero p {margin:.48rem 0 0;color:#e9f7f3;font-size:1.02rem;max-width:820px;}
  .ccg-card-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1rem;align-items:stretch;}
  .ccg-card {background:linear-gradient(180deg,#fff,var(--ccg-soft));border:1px solid var(--ccg-line);
    border-radius:15px;padding:1rem 1.05rem;min-height:166px;box-shadow:0 5px 16px #1733310a;
    display:flex;flex-direction:column;height:100%;overflow:visible;}
  .ccg-card h4 {color:#0c544f;margin:0 0 .55rem;min-height:3.15em;line-height:1.45;
    font-size:1.08rem;overflow:visible;overflow-wrap:normal!important;word-break:normal!important;hyphens:none!important;}
  .ccg-card p{font-size:.88rem;margin:auto 0 0;color:#405656;line-height:1.45;overflow:visible;
    overflow-wrap:normal!important;word-break:normal!important;hyphens:none!important;}
  .ccg-ready-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.65rem;}
  .ccg-ready-item {padding:.55rem .7rem;border:1px solid var(--ccg-line);border-radius:10px;line-height:1.35;overflow-wrap:anywhere;}
  .ccg-progress-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.65rem;}
  .ccg-step {border-left:4px solid var(--ccg-line);padding:.48rem .7rem;margin:.25rem 0;background:#fff;border-radius:0 10px 10px 0;}
  .ccg-step.done {border-left-color:#2b9a70;background:#f2faf6}.ccg-step.pending{color:var(--ccg-muted)}
  .ccg-class {border:1px solid var(--ccg-line);border-radius:12px;padding:.45rem;margin-bottom:.6rem;}
  div[data-testid="stMetric"] {background:var(--ccg-soft);border:1px solid var(--ccg-line);border-radius:13px;padding:.62rem .78rem;}
  div[data-testid="stMetricValue"] {font-size:clamp(1.2rem,2.3vw,1.72rem);white-space:normal;overflow-wrap:anywhere;line-height:1.18;}
  div[data-testid="stButton"] button[kind="primary"] {background:var(--ccg-primary);border-color:var(--ccg-primary);}
  [data-testid="stSidebar"] label p, [data-testid="stSidebar"] [role="radiogroup"] p,
  div[data-baseweb="select"] span {white-space:normal!important;overflow:visible!important;text-overflow:clip!important;line-height:1.35!important;}
  @media (max-width:1100px) {.ccg-card-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.ccg-ready-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
  @media (max-width:760px) {.block-container{padding:3.25rem .85rem 2rem!important}.ccg-card-grid,.ccg-ready-grid,.ccg-progress-grid{grid-template-columns:1fr}.ccg-card{min-height:auto}.ccg-card h4{min-height:auto}.ccg-hero{padding:1.2rem}.ccg-hero p{font-size:.92rem}}
</style>""", unsafe_allow_html=True)


def _config_preview(values: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        config = build_config(values)
    except ValueError as exc:
        return None, str(exc)
    return config, yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _recommendation_preview_input_audit(paths: list[Path]) -> list[dict[str, Any]]:
    """Describe every ligand file used by the chemistry preview."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        row: dict[str, Any] = {"requested_path": str(path), "exists": path.exists()}
        if not path.exists():
            row.update({"status": "missing_file", "hetatm_records": 0, "conect_records": 0})
            rows.append(row)
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            row.update({"status": f"unreadable: {exc}", "hetatm_records": 0, "conect_records": 0})
            rows.append(row)
            continue
        hetatm_count = sum(line.startswith("HETATM") for line in lines)
        conect_count = sum(line.startswith("CONECT") for line in lines)
        row.update({
            "status": "ready" if hetatm_count else "no_hetatm_records",
            "hetatm_records": hetatm_count, "conect_records": conect_count,
        })
        rows.append(row)
    return rows


def _build_recommendation_preview(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Generate an audited chemistry-recommendation preview."""
    config = normalize_project_config(config)
    project = config.get("project", {}) or {}
    target_id = str(project.get("target_id", "target"))
    output_root = _repository_path(project.get("output_root", f"results/{target_id}"))
    chemistry = config.get("step09", {}) or {}
    chemistry_dir = output_root / public_output_subdir(
        chemistry.get("output_subdir"), "08_chemical_equivalence"
    )
    recommendation_table = chemistry_dir / "tables" / f"{target_id}_step08_ligand_unit_recommendations.csv"
    inventory_table = chemistry_dir / "tables" / f"{target_id}_step08_substrate_inventory.csv"
    configured_target = validated_contact_unit_target(target_id)
    if recommendation_table.exists() and not configured_target:
        import pandas as pd
        recommendations = pd.read_csv(recommendation_table).fillna("")
        inventory = pd.read_csv(inventory_table).fillna("") if inventory_table.exists() else pd.DataFrame()
        return (
            recommendations.to_dict(orient="records"),
            inventory.to_dict(orient="records"),
            {
                "preview_source": "completed_step08",
                "recommendation_table": str(recommendation_table),
                "n_ligand_unit_recommendations": int(len(recommendations)),
            },
            [],
        )

    inputs = config.get("inputs", {}) or {}
    raw_paths = inputs.get("ligand_pdbs") or [inputs.get("ligand_pdb", "")]
    ligand_paths = [
        Path(str(path)) if Path(str(path)).is_absolute() else REPOSITORY_ROOT / str(path)
        for path in raw_paths if str(path).strip()
    ]
    input_audit = _recommendation_preview_input_audit(ligand_paths)
    existing_paths = [path for path in ligand_paths if path.exists()]
    from catcongraph.steps.step09_chemical_equivalence import (
        build_ligand_unit_recommendations, build_substrate_inventory,
    )
    recommendations, _mapping, report = build_ligand_unit_recommendations(
        config, existing_paths, target_id=target_id
    )
    inventory = build_substrate_inventory(config, existing_paths)
    report = dict(report)
    report.update({
        "preview_source": (
            "configured_suggestions"
            if configured_target
            else "template_ligand_preliminary"
        ),
        "requested_ligand_paths": [str(path) for path in ligand_paths],
        "existing_ligand_paths": [str(path) for path in existing_paths],
        "missing_ligand_paths": [str(path) for path in ligand_paths if not path.exists()],
        "n_preview_inventory_rows": int(len(inventory)),
    })
    return (
        recommendations.to_dict(orient="records"),
        inventory.to_dict(orient="records"), report, input_audit,
    )


def _input_path_help(st: Any) -> None:
    st.caption(_t(st,
        "路径在运行 AF2-CatGraph 的服务器/电脑上解释；相对路径相对于软件仓库，绝对路径也可用。",
        "Paths are interpreted on the computer/server running AF2-CatGraph. Relative paths use the repository root; absolute paths also work."))


def _repository_path(value: Any) -> Path:
    """Resolve a GUI path for presentation-only readiness checks."""
    path = Path(str(value or "")).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _path_exists(value: Any, *, directory: bool = False) -> bool:
    if not str(value or "").strip():
        return False
    path = _repository_path(value)
    return path.is_dir() if directory else path.is_file()


def _project_readiness(values: dict[str, Any]) -> tuple[int, list[tuple[str, bool]]]:
    ligand_inputs = split_paths(values.get("ligand_paths"))
    checks = [
        ("AF2/ColabFold PDB", _path_exists(values.get("af2_dir"), directory=True)),
        ("Template protein", _path_exists(values.get("template_protein_pdb"))),
        ("Ligand inputs", bool(ligand_inputs) and all(_path_exists(item) for item in ligand_inputs)),
        ("ConSurf", _path_exists(values.get("conservation_file"))),
        ("GED", _path_exists(values.get("ged_executable"))),
    ]
    return sum(ok for _, ok in checks), checks


def _workflow_progress(values: dict[str, Any]) -> list[tuple[str, bool]]:
    root = _resolve_output_root(values)
    directories = [
        ("01", "01_af2_prepare"), ("02", "02_template_guided_docking"),
        ("03", "03_clash_filter"), ("04", "04_catalytic_region_plddt_qc"),
        ("05", "05_plip_interaction_graphs"), ("06", "06_key_residue_screening"),
        ("07", "07_ged_classification"), ("08", "08_chemical_equivalence"),
        ("09", "09_class_visualization"),
    ]
    return [(number, (root / folder).exists()) for number, folder in directories]


def _render_overview(st: Any) -> None:
    values = _values(st)
    st.markdown(f"""<div class="ccg-hero"><h1>AF2-CatGraph</h1><p>{_t(st, '从构象集合到相互作用类别、候选构象和可视化结果。', 'From structural ensembles to interaction classes, candidate conformations, and visual results.')}</p></div>""", unsafe_allow_html=True)
    ready, readiness = _project_readiness(values)
    progress = _workflow_progress(values)
    completed = sum(done for _, done in progress)
    c1, c2, c3 = st.columns(3)
    c1.metric(_t(st, "必要输入就绪", "Required inputs ready"), f"{ready}/5")
    c2.metric(_t(st, "已生成步骤目录", "Step folders found"), f"{completed}/9")
    c3.metric(_t(st, "工作流程", "Workflow"), _t(st, "固定 01–09 步", "Fixed Steps 01–09"))
    with st.container(border=True):
        st.markdown("#### " + _t(st, "项目就绪状态", "Project readiness"))
        ready_items = "".join(
            f'<div class="ccg-ready-item">{"✓" if ok else "○"} {label}</div>'
            for label, ok in readiness
        )
        st.markdown(f'<div class="ccg-ready-grid">{ready_items}</div>', unsafe_allow_html=True)
        if ready < 5:
            st.caption(_t(st, "在“项目设置”补全带圆圈的输入，然后运行 Preflight。", "Complete the circled inputs in Project setup, then run Preflight."))
    cards = [
        ("00", _t(st, "运行前检查", "Preflight"), _t(st, "检查路径、依赖和模板兼容性。", "Validate paths, dependencies, and template compatibility.")),
        ("01–03", _t(st, "准备、放置与筛选", "Prepare, place & filter"), _t(st, "全局 pLDDT、模板引导放置与 VDW 碰撞筛选。", "Global pLDDT, template-guided placement, and VDW-clash filtering.")),
        ("04–05", _t(st, "局部质量与相互作用", "Local quality & interactions"), _t(st, "检查催化区域 pLDDT，并生成 PLIP 相互作用图。", "Check catalytic-region pLDDT and build PLIP interaction graphs.")),
        ("06–08", _t(st, "残基、拓扑与接触单元", "Residues, topology & contact units"), _t(st, "筛选关键残基、执行 GED 分类，并按所选路线进行配体接触单元分组。", "Screen key residues, classify with GED, and group ligand contact units under the selected route.")),
        ("09", _t(st, "结果与可视化", "Results & visualization"), _t(st, "输出类别图、汇总表、候选结构和三维视图。", "Export class plots, summaries, candidates, and 3D views.")),
    ]
    card_html = "".join(
        f'<div class="ccg-card"><h4>{number} · {title}</h4><p>{body}</p></div>'
        for number, title, body in cards
    )
    st.markdown(f'<div class="ccg-card-grid">{card_html}</div>', unsafe_allow_html=True)
    st.subheader(_t(st, "主流程进度", "Main-workflow progress"))
    progress_names = [_t(st, zh, en) for zh, en in [
        ("全局 pLDDT", "Global pLDDT"), ("模板引导放置", "Template placement"),
        ("碰撞筛选", "Clash filter"), ("局部 pLDDT", "Local pLDDT"),
        ("PLIP", "PLIP"), ("关键残基", "Key residues"),
        ("GED 分类", "GED classification"), ("接触单元", "Contact units"),
        ("汇总与可视化", "Summary & visualization"),
    ]]
    progress_html = "".join(
        f'<div class="ccg-step {"done" if done else "pending"}"><strong>{number} · {name}</strong><br>{_t(st, "完成" if done else "待运行", "Done" if done else "Pending")}</div>'
        for (number, done), name in zip(progress, progress_names)
    )
    st.markdown(f'<div class="ccg-progress-grid">{progress_html}</div>', unsafe_allow_html=True)
    st.subheader(_t(st, "推荐下一步", "Recommended next step"))
    st.write(_t(st,
        "在“项目设置”保存配置。先运行 Preflight；随后用 3–5 个结构试运行 Step01–09，核查中间结构和表格后再运行全量。",
        "Save a configuration in Project setup. Run Preflight first; then pilot Step01–09 on 3–5 structures and inspect intermediate structures and tables before processing the full ensemble."))


def _render_colabfold(st: Any) -> None:
    values = _values(st)
    st.subheader(_t(st, "步骤 0 · 可选：本地 ColabFold", "Step 0 · Optional local ColabFold"))
    st.caption(_t(st, "公共 Step01–09 只需要 protein-only PDB。本页仅启动本地 ColabFold，不改后续流程。", "Public Step01–09 require protein-only PDBs. This page only launches local ColabFold and does not alter later steps."))
    source = st.radio(_t(st, "结构来源", "Structure source"), [_t(st, "已有 PDB（推荐）", "Existing PDBs (recommended)"), _t(st, "运行本地 ColabFold", "Run local ColabFold")], horizontal=True)
    if source.startswith("已有") or source.startswith("Existing"):
        st.success(_t(st, "请在“项目设置”填写 AF2/ColabFold PDB 文件夹。", "Set the AF2/ColabFold PDB folder in Project setup."))
        return
    with st.form("colabfold_form"):
        a, b = st.columns(2)
        fasta = a.text_input(_t(st, "FASTA 输入文件", "FASTA input"), values.get("colabfold_fasta", "data/MY_ENZYME/input.fasta"))
        executable = b.text_input(_t(st, "ColabFold 可执行程序", "ColabFold executable"), values.get("colabfold_executable", "colabfold_batch"))
        a, b = st.columns(2)
        conda_executable = a.text_input(_t(st, "Conda 可执行程序", "Conda executable"), values.get("conda_executable", "conda"))
        colabfold_env = b.text_input(_t(st, "ColabFold 专属 Conda 环境", "ColabFold Conda environment"), values.get("colabfold_conda_env", ""), help=_t(st, "留空时在当前环境直接运行。", "Leave blank to run in the current environment."))
        a, b, c = st.columns(3)
        seeds = a.number_input(_t(st, "随机种子数", "Number of seeds"), 1, 10000, int(values.get("colabfold_seeds", 64)))
        models = b.number_input(_t(st, "模型数", "Number of models"), 1, 5, int(values.get("colabfold_models", 5)))
        recycles = c.number_input(_t(st, "Recycle 次数", "Recycles"), 0, 48, int(values.get("colabfold_recycles", 3)))
        a, b, c = st.columns(3)
        clusters = a.number_input("max MSA clusters", 1, 512, int(values.get("colabfold_msa_clusters", 16)))
        extra = b.number_input("max extra MSA", 0, 4096, int(values.get("colabfold_extra_msa", 32)))
        dropout = c.checkbox(_t(st, "启用 dropout", "Use dropout"), bool(values.get("colabfold_dropout", True)))
        apply = st.form_submit_button(_t(st, "应用设置", "Apply settings"))
    updates = {"colabfold_fasta": fasta, "colabfold_executable": executable, "conda_executable": conda_executable, "colabfold_conda_env": colabfold_env, "colabfold_seeds": seeds, "colabfold_models": models, "colabfold_recycles": recycles, "colabfold_msa_clusters": clusters, "colabfold_extra_msa": extra, "colabfold_dropout": dropout}
    if apply:
        _set_values(st, updates); values = _values(st)
        st.rerun()
    command = _colabfold_command(values)
    with st.expander(_t(st, "查看将执行的命令", "Show command")):
        st.code(" ".join(command), language="bash")
    if st.button(_t(st, "启动 ColabFold", "Start ColabFold"), type="primary", disabled=_job_running(st)):
        _start_job(st, command); st.rerun()


def _colabfold_command(values: dict[str, Any]) -> list[str]:
    command = [str(values.get("colabfold_executable", "colabfold_batch")), str(values.get("colabfold_fasta", "")), str(values.get("af2_dir", "")), "--model-type", "alphafold2_ptm", "--num-models", str(int(values.get("colabfold_models", 5))), "--num-recycle", str(int(values.get("colabfold_recycles", 3))), "--num-seeds", str(int(values.get("colabfold_seeds", 64))), "--max-msa", f"{int(values.get('colabfold_msa_clusters', 16))}:{int(values.get('colabfold_extra_msa', 32))}"] + (["--use-dropout"] if values.get("colabfold_dropout", True) else [])
    return _conda_wrapped_command(command, values.get("conda_executable", "conda"), values.get("colabfold_conda_env", ""))


def _conda_wrapped_command(command: list[str], conda_executable: Any, environment_name: Any) -> list[str]:
    """Run a command in a named Conda environment without mutating the GUI shell."""
    environment = str(environment_name or "").strip()
    if not environment:
        return command
    if command and Path(command[0]).resolve() == Path(sys.executable).resolve():
        command = ["python", *command[1:]]
    return [str(conda_executable or "conda"), "run", "-n", environment, "--no-capture-output", *command]


def _af2_preprocess_command(
    input_dir: str, output_dir: str, prefix: str, residue_ranges: str, *,
    chains: str = "", recursive: bool = True, rank_start: int = 1,
    conda_executable: str = "conda", workflow_conda_env: str = "",
) -> list[str]:
    """Build the standalone, source-preserving AF2 preprocessing command."""
    command = [
        sys.executable, str(REPOSITORY_ROOT / "scripts" / "00_af2_trim_and_rename.py"),
        "--input-dir", input_dir, "--output-dir", output_dir, "--name", prefix,
        "--remove", residue_ranges, "--rank-start", str(rank_start),
    ]
    if chains.strip():
        command += ["--chains", chains.strip()]
    if not recursive:
        command.append("--no-recursive")
    return _conda_wrapped_command(command, conda_executable, workflow_conda_env)


def _render_af2_preprocess(st: Any) -> None:
    """Expose the existing AF2 trimming utility in Studio without changing it."""
    values = _values(st)
    st.subheader(_t(st, "步骤 0A · AF2/ColabFold 结构预处理", "Step 0A · AF2/ColabFold preprocessing"))
    st.caption(_t(
        st,
        "此工具复制到新文件夹：删除指定残基区间、保留其余残基原编号、连续命名为 <名称>.AF2rank1/2/...，并输出 CSV 审计表。原始 AF2 PDB 不会被改写。",
        "This tool writes a new folder: it removes selected residue intervals, keeps the original numbering of retained residues, renames files as <name>.AF2rank1/2/..., and writes a CSV audit. Source AF2 PDBs are never changed.",
    ))
    _input_path_help(st)
    default_input = str(values.get("af2_dir", ""))
    default_output = f"{default_input.rstrip('/')}__trimmed" if default_input else "data/MY_ENZYME/01_af2_trimmed"
    with st.form("af2_preprocess_form", clear_on_submit=False):
        input_dir = st.text_input(_t(st, "原始 AF2 PDB 文件夹", "Source AF2 PDB folder"), default_input)
        output_dir = st.text_input(_t(st, "新的预处理输出文件夹", "New preprocessed output folder"), default_output)
        a, b = st.columns(2)
        prefix = a.text_input(_t(st, "输出名称前缀", "Output name prefix"), str(values.get("target_id", "MY_ENZYME")), help=_t(st, "例如 6B90；将输出 6B90.AF2rank1.pdb。", "For example 6B90; output is 6B90.AF2rank1.pdb."))
        ranges = b.text_input(_t(st, "删除的残基区间", "Residue intervals to remove"), "1-120,290-300", help=_t(st, "闭区间；例如 1-120,290-300，或单个残基 415。", "Inclusive intervals, e.g. 1-120,290-300, or a single residue 415."))
        a, b, c = st.columns(3)
        chains = a.text_input(_t(st, "仅处理链（可选）", "Chains only (optional)"), "", help=_t(st, "留空表示全部链；多个链用逗号。", "Blank means every chain; separate chains with commas."))
        rank_start = b.number_input(_t(st, "输出 rank 起始值", "First output rank"), 1, 100000, 1)
        recursive = c.checkbox(_t(st, "搜索子文件夹", "Include subfolders"), True)
        submit = st.form_submit_button(_t(st, "开始预处理", "Start preprocessing"), type="primary")
    command = _af2_preprocess_command(
        input_dir, output_dir, prefix, ranges, chains=chains,
        recursive=recursive, rank_start=int(rank_start),
        conda_executable=str(values.get("conda_executable", "conda")),
        workflow_conda_env=str(values.get("workflow_conda_env", "")),
    )
    with st.expander(_t(st, "查看将执行的命令", "Show command")):
        st.code(" ".join(command), language="bash")
    if submit:
        try:
            # Validate the fields early, while the worker still owns all file writes.
            from catcongraph.structure.af2_preprocess import parse_residue_ranges
            parse_residue_ranges(ranges)
            if not input_dir.strip() or not output_dir.strip() or not prefix.strip():
                raise ValueError(_t(st, "请填写输入目录、输出目录和名称前缀。", "Enter input folder, output folder, and name prefix."))
            _start_job(st, command)
            st.session_state.ccg_af2_preprocess_output = output_dir.strip()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    job = st.session_state.get("ccg_job")
    finished_output = str(st.session_state.get("ccg_af2_preprocess_output", ""))
    if finished_output and job and not job.running and job.returncode == 0:
        st.success(_t(st, "预处理完成。确认审计表后，可将该新文件夹作为主流程的 AF2 输入。", "Preprocessing completed. After reviewing the audit table, use this new folder as the main workflow's AF2 input."))
        if st.button(_t(st, "将预处理文件夹设为当前 AF2 输入", "Use preprocessed folder as current AF2 input")):
            _set_values(st, {"af2_dir": finished_output})
            st.success(_t(st, "已更新当前项目设置；请在“项目设置”保存 YAML。", "Current project settings updated; save the YAML in Project setup."))
    st.divider()
    _render_job_panel(st)


def _render_project_setup(st: Any) -> None:
    values = _values(st)
    st.subheader(_t(st, "项目设置", "Project setup"))
    st.caption(_t(st, "输入框支持原生选择、Ctrl+C 和 Ctrl+V 覆盖；点击“应用”后才写入当前配置。", "Fields support native selection and Ctrl+C/Ctrl+V replacement; values enter the current configuration only after Apply."))
    c1, c2 = st.columns([1, 3])
    existing = c2.text_input(_t(st, "读取已有 YAML（可选）", "Load existing YAML (optional)"), "", key="ccg_existing_config")
    if c1.button(_t(st, "读取配置", "Load configuration")):
        try:
            st.session_state.ccg_values = config_to_values(existing)
            st.session_state.ccg_config_path = existing
            _clear_project_widget_state(st)
            st.session_state.pop("ccg_save_config", None)
            st.session_state.pop("ccg_run_config", None)
            st.rerun()
        except Exception as exc:
            st.error(_t(st, f"无法读取配置：{exc}", f"Could not load configuration: {exc}"))
    with st.container(border=True):
        st.markdown("#### " + _t(st, "全局重命名", "Global rename"))
        rename_col, button_col = st.columns([3, 1])
        new_target_id = rename_col.text_input(
            _t(st, "新项目 ID", "New project ID"),
            str(values.get("target_id", "MY_ENZYME")),
            key=_project_widget_key(st, "ccg_global_rename_id"),
        )
        if button_col.button(_t(st, "一键应用", "Apply everywhere"), type="primary", use_container_width=True):
            try:
                old_target_id = str(values.get("target_id", "")).strip()
                renamed = global_rename_values(values, new_target_id)
                st.session_state.ccg_values = renamed
                old_config_path = str(st.session_state.get("ccg_config_path", "")).strip()
                new_config_path = str(global_rename_config_path(
                    old_config_path, old_target_id, new_target_id, REPOSITORY_ROOT
                ))
                st.session_state.ccg_config_path = new_config_path
                _clear_project_widget_state(st)
                st.session_state["ccg_save_config"] = new_config_path
                st.session_state["ccg_run_config"] = new_config_path
                st.success(_t(st, "项目名及由旧项目名派生的路径已同步。", "Project name and target-derived paths were synchronized."))
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        st.caption(_t(st, "此操作更新下一次保存和运行所用的名称与路径；不会移动或重命名磁盘上的既有结果。", "This updates names and paths for the next save/run; existing files on disk are not moved or renamed."))
    _input_path_help(st)
    with st.form("project_form", clear_on_submit=False):
        a, b = st.columns(2)
        target_id = a.text_input(_t(st, "项目 ID *", "Project ID *"), str(values["target_id"]), key=_project_widget_key(st, "ccg_field_target_id"), help=_t(st, "仅允许字母、数字、点、下划线和连字符。", "Letters, numbers, '.', '_' and '-' only."))
        output_root = b.text_input(_t(st, "结果目录 *", "Output directory *"), str(values["output_root"]), key=_project_widget_key(st, "ccg_field_output_root"))
        af2_dir = st.text_input(_t(st, "AF2/ColabFold protein-only PDB 文件夹 *", "AF2/ColabFold protein-only PDB folder *"), str(values["af2_dir"]), key=_project_widget_key(st, "ccg_field_af2_dir"))
        template = st.text_input(_t(st, "模板复合物的蛋白 PDB *", "Template-complex protein PDB *"), str(values["template_protein_pdb"]), key=_project_widget_key(st, "ccg_field_template"))
        ligands = st.text_area(_t(st, "配体 / 底物 / 辅因子 / 金属 PDB *", "Ligand / substrate / cofactor / metal PDB *"), str(values["ligand_paths"]), key=_project_widget_key(st, "ccg_field_ligands"), height=92, help=_t(st, "一行一个文件。顺序即 Step02 顺序放置（dock）的顺序。", "One file per line. This is the sequential placement (dock) order in Step02."))
        a, b = st.columns(2)
        conservation = a.text_input(
            _t(st, "保守性文件（ConSurf ZIP / TAR.GZ / GZ / CSV/TSV）*", "Conservation file (ConSurf ZIP / TAR.GZ / GZ / CSV/TSV) *"),
            str(values["conservation_file"]),
            key=_project_widget_key(st, "ccg_field_conservation"),
            help=_t(
                st,
                "ConSurf 网站直接下载、名为 new1.gz 的结果包可原样填写；无需手动解压。",
                "A ConSurf website download named new1.gz is accepted unchanged; do not unpack it manually.",
            ),
        )
        ged = b.text_input(_t(st, "Graph_Edit_Distance ged 可执行文件 *", "Graph_Edit_Distance ged executable *"), str(values["ged_executable"]), key=_project_widget_key(st, "ccg_field_ged"))
        resnames = st.text_input(_t(st, "分析用配体残基名 *", "Ligand residue name(s) for analysis *"), str(values["ligand_resnames"]), key=_project_widget_key(st, "ccg_field_resnames"), help=_t(st, "多个实体以英文逗号分隔，例如 ATP, MG, SUB。", "Separate entities with commas, for example ATP, MG, SUB."))
        st.markdown("#### " + _t(st, "常用质量控制参数", "Common quality-control parameters"))
        a, b, c, d = st.columns(4)
        global_plddt = a.number_input(_t(st, "Step01 全局 pLDDT 下限", "Step01 global pLDDT minimum"), 0.0, 100.0, float(values["global_plddt_min"]), 1.0)
        clash_max = b.number_input(_t(st, "Step03 最大碰撞数", "Step03 maximum clashes"), 0, 1000, int(values["clash_max"]))
        mean_plddt = c.number_input(_t(st, "Step04 局部平均 pLDDT", "Step04 local mean pLDDT"), 0.0, 100.0, float(values["catalytic_mean_plddt_min"]), 1.0)
        p10_plddt = d.number_input(_t(st, "Step04 局部 p10 pLDDT", "Step04 local p10 pLDDT"), 0.0, 100.0, float(values["catalytic_p10_plddt_min"]), 1.0)
        # Score-scope and manifest controls remain supported in YAML but are
        # intentionally hidden from the compact scientific form.  Keeping the
        # loaded values here prevents a save from resetting an older project.
        plddt_filter_scope = str(values.get("plddt_filter_scope", "source_if_available"))
        af2_preprocess_manifest = str(values.get("af2_preprocess_manifest", "auto"))
        apply = st.form_submit_button(_t(st, "应用项目设置", "Apply project settings"), type="primary")
    if apply:
        _set_values(st, {"target_id": target_id, "output_root": output_root, "af2_dir": af2_dir, "template_protein_pdb": template, "ligand_paths": ligands, "conservation_file": conservation, "ged_executable": ged, "ligand_resnames": resnames, "global_plddt_min": global_plddt, "plddt_filter_scope": plddt_filter_scope, "af2_preprocess_manifest": af2_preprocess_manifest, "clash_max": clash_max, "catalytic_mean_plddt_min": mean_plddt, "catalytic_p10_plddt_min": p10_plddt})
        st.rerun()
    _render_consurf_input_check(st)
    _render_config_export(st)


def _render_consurf_input_check(st: Any) -> None:
    """Let GUI users verify website downloads before saving/running a project."""
    values = _values(st)
    with st.expander(_t(st, "检查 ConSurf 保守性输入", "Check ConSurf conservation input")):
        st.caption(_t(
            st,
            "支持旧版 ZIP、网站下载的 TAR.GZ/GZ（例如 new1.gz）、单一 gzip 文本及 CSV/TSV。网站评分表中的 PDB 残基编号会优先保留。",
            "Supports legacy ZIP, website TAR.GZ/GZ downloads (for example new1.gz), single gzip text files, and CSV/TSV. PDB residue numbering embedded in website grade tables is retained preferentially.",
        ))
        source_text = str(values.get("conservation_file", "")).strip()
        if st.button(_t(st, "解析并检查当前文件", "Parse and check current file"), key="ccg_check_consurf"):
            if not source_text:
                st.error(_t(st, "请先填写保守性文件路径。", "Enter a conservation-file path first."))
                return
            source_path = Path(source_text)
            if not source_path.is_absolute():
                source_path = REPOSITORY_ROOT / source_path
            try:
                from catcongraph.steps.step07_key_residue_screening import load_consurf_scores
                scores, metadata = load_consurf_scores(source_path, {})
                st.success(_t(st, f"已解析 {len(scores)} 个 ConSurf 残基。", f"Parsed {len(scores)} ConSurf residues."))
                st.json(metadata)
                preview_columns = [
                    column for column in [
                        "consurf_position", "pdb_residue_annotation", "residue_key",
                        "consurf_grade", "consurf_score", "residue_mapping_source",
                    ] if column in scores.columns
                ]
                st.dataframe(scores[preview_columns].head(12), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(_t(st, f"无法解析 ConSurf 输入：{exc}", f"Could not parse ConSurf input: {exc}"))


def _render_config_export(st: Any) -> None:
    values = _values(st)
    suggested = default_config_path({"target_id": values.get("target_id")}, REPOSITORY_ROOT)
    config_path = st.text_input(_t(st, "保存 YAML 到", "Save YAML to"), st.session_state.ccg_config_path or str(suggested), key="ccg_save_config")
    st.session_state.ccg_config_path = config_path
    config, preview = _config_preview(values)
    if config is None:
        st.error(preview); return
    with st.expander(_t(st, "预览将要运行的 YAML", "Preview YAML to be run")):
        st.code(preview, language="yaml")
    a, b = st.columns(2)
    if a.button(_t(st, "保存配置", "Save configuration"), type="primary"):
        try:
            st.session_state.ccg_config_path = str(write_config(config_path, config))
            st.success(_t(st, f"已保存：{config_path}", f"Saved: {config_path}"))
        except OSError as exc:
            st.error(str(exc))
    b.download_button(_t(st, "下载 YAML", "Download YAML"), preview, f"{values['target_id']}_af2_catgraph.yaml", "text/yaml")


def _render_advanced(st: Any) -> None:
    values = _values(st)
    st.subheader(_t(st, "步骤参数与配体接触单元分组", "Step parameters and ligand contact-unit grouping"))
    st.info(_t(
        st,
        "默认值对应 AF2-CatGraph 的标准计算流程。参数可以调整；任何修改都会明确写入 YAML，因此旧配置不加载、不修改时仍按原设置运行。",
        "Defaults follow the standard AF2-CatGraph workflow. Parameters remain editable; every change is written explicitly to YAML, so an unchanged existing config retains its original settings.",
    ))
    with st.form("advanced_form", clear_on_submit=False):
        with st.expander("Step01–02 · " + _t(st, "构象筛选与模板引导放置", "ensemble filtering and template-guided placement"), expanded=True):
            a, b = st.columns(2)
            alignment_options = ["auto", "sequence_aware"]
            current_alignment = str(values.get("step02_alignment_mode", "auto"))
            visible_alignment_index = (
                alignment_options.index(current_alignment)
                if current_alignment in alignment_options else None
            )
            alignment = a.selectbox(
                _t(st, "蛋白对齐方式", "Protein-alignment method"), alignment_options,
                index=visible_alignment_index,
                placeholder=_t(st, "保留已有兼容设置", "Preserve stored compatibility setting"),
                format_func=lambda item: {
                    "auto": _t(st, "自动（保持既有配置行为）", "Automatic (preserve established behavior)"),
                    "sequence_aware": _t(st, "序列感知 Cα 对齐", "Sequence-aware Cα alignment"),
                }[item],
            )
            multi = b.selectbox(
                _t(st, "多底物放置", "Multiple-component placement"),
                ["sequential", "rigid_assembly"],
                index=["sequential", "rigid_assembly"].index(str(values.get("step02_multi_ligand_mode", "sequential"))),
                format_func=lambda item: _t(st, "按输入列表逐个放置（推荐）", "Place sequentially in input order (recommended)") if item == "sequential" else _t(st, "作为整体刚体放置", "Place as one rigid assembly"),
            )
            st.text_input(
                _t(st, "Step02 输入来源", "Step02 input source"),
                _t(st, "Step01 筛选通过的结构", "Structures passed by Step01"),
                disabled=True,
            )
            if current_alignment not in alignment_options or str(values.get("step02_input_source", "step01")) != "step01":
                st.caption(_t(
                    st,
                    "此项目包含已隐藏的兼容设置；若不选择新的对齐方式，保存时将保持原值。",
                    "This project contains hidden compatibility settings; saving preserves them unless a visible alignment method is selected.",
                ))
            alignment = alignment or current_alignment
            input_source = str(values.get("step02_input_source", "step01"))
            identity = float(values["min_sequence_identity"])
            coverage = float(values["min_alignment_coverage"])
            matched_ca = int(values["min_matched_ca"])
            overwrite = bool(values["step02_overwrite"])

        with st.expander("Step03–04 · " + _t(st, "碰撞与催化区域质量控制", "clash and catalytic-region quality control")):
            a, b, c = st.columns(3)
            clash_max = a.number_input(_t(st, "最大碰撞数", "Maximum clash count"), 0, 1000, int(values["clash_max"]))
            clash_delta = b.number_input("VDW delta (Å)", 0.0, 3.0, float(values["clash_delta"]), 0.1)
            cutoff = c.number_input(_t(st, "催化区域半径 (Å)", "Catalytic-region radius (Å)"), 1.0, 30.0, float(values["catalytic_distance_cutoff"]), 0.5)
            include_names = str(values.get("step03_include_resnames", ""))
            a, b = st.columns(2)
            mean_plddt = a.number_input(_t(st, "局部平均 pLDDT 下限", "Local mean pLDDT minimum"), 0.0, 100.0, float(values["catalytic_mean_plddt_min"]), 1.0)
            p10_plddt = b.number_input(_t(st, "局部 p10 pLDDT 下限", "Local p10 pLDDT minimum"), 0.0, 100.0, float(values["catalytic_p10_plddt_min"]), 1.0)

        with st.expander("Step05–06 · PLIP / " + _t(st, "关键残基", "key residues")):
            plip_exe = st.text_input(_t(st, "PLIP 可执行程序", "PLIP executable"), str(values["plip_executable"]))
            a, b, c = st.columns(3)
            consurf = a.number_input(_t(st, "最低保守性等级", "Minimum conservation grade"), 1, 9, int(values["consurf_grade_min"]))
            residue_distance = b.number_input(_t(st, "残基–底物距离上限 (Å)", "Residue–ligand distance maximum (Å)"), 0.5, 15.0, float(values["residue_ligand_distance_max"]), 0.1)
            occurrence = c.number_input(_t(st, "最低出现频率", "Minimum occurrence frequency"), 0.0, 1.0, float(values["occurrence_frequency_min"]), 0.05)
            st.caption(_t(st, "关键残基仍需同时满足：保守性、距离接触、距离出现频率和 PLIP 接触频率。", "A key residue must still satisfy conservation, distance contact, distance frequency, and PLIP-contact frequency."))
            node_id = str(values.get("plip_ligand_node_id", "residue_aware"))
            connectivity = str(values.get("plip_connectivity_source", "auto"))
            plip_run = bool(values["plip_run"])
            plip_save_intermediate = bool(values["plip_save_intermediate"])
            require_graph_occurrence = bool(values["require_graph_occurrence"])
            mapping_enabled = bool(values["consurf_mapping_enabled"])
            mapping_target = float(values["consurf_mapping_min_target_coverage"])
            mapping_contact = float(values["consurf_mapping_min_contact_coverage"])
            mapping_fail = bool(values["consurf_mapping_fail_on_low_coverage"])

        with st.expander("Step07 · " + _t(st, "图编辑距离分类", "graph-edit-distance classification")):
            ged_threshold = st.number_input(_t(st, "GED 阈值", "GED threshold"), 0.0, 100.0, float(values["ged_threshold"]), 1.0)
            workers = int(values["ged_max_workers"])
            timeout = int(values["ged_timeout_seconds"])
            same_edges = bool(values["ged_require_same_edge_count"])

        with st.expander("Step08 · " + _t(st, "配体接触单元分组与候选类别", "ligand contact-unit grouping and candidate classes"), expanded=True):
            a, b, c = st.columns([1, 1, 2])
            common = a.number_input(_t(st, "高频相互作用下限", "Common-interaction minimum"), 0.0, 1.0, float(values["common_frequency_min"]), 0.05)
            rare = b.number_input(_t(st, "低频相互作用上限", "Rare-interaction maximum"), 0.0, 1.0, float(values["rare_frequency_max"]), 0.05)
            contact_unit_options = [CONTACT_UNIT_UNIVERSAL, CONTACT_UNIT_CONSERVATIVE]
            current_contact_mode = str(values.get("contact_unit_mode", CONTACT_UNIT_UNIVERSAL))
            visible_contact_index = (
                contact_unit_options.index(current_contact_mode)
                if current_contact_mode in contact_unit_options else None
            )
            selected_contact_mode = c.selectbox(
                _t(st, "分析路线", "Analysis route"),
                contact_unit_options,
                index=visible_contact_index,
                placeholder=_t(st, "保留已有兼容设置", "Preserve stored compatibility setting"),
                format_func=lambda item: (
                    _t(st, "S4 通用配体接触单元分组（推荐）", "Universal S4 ligand contact-unit grouping (recommended)")
                    if item == CONTACT_UNIT_UNIVERSAL
                    else _t(st, "保留原子级 GED 类别", "Retain atom-level GED classes")
                ),
                help=_t(
                    st,
                    "两条路线使用相同的 GED、相互作用频率统计和候选筛选；区别仅在于是否进行配体接触单元分组。",
                    "Both routes use the same GED, interaction-frequency analysis, and candidate screening; they differ only in whether ligand contact-unit grouping is applied.",
                ),
            )
            contact_unit_mode = selected_contact_mode or current_contact_mode
            if contact_unit_mode == CONTACT_UNIT_UNIVERSAL:
                st.info(_t(
                    st,
                    "程序对所有酶应用同一套 S4 规则，随后进行相互作用标签分组、类别合并、频率分析和候选筛选。",
                    "The same S4 rules are applied to every enzyme, followed by interaction-label grouping, class merging, frequency analysis, and candidate screening.",
                ))
            elif contact_unit_mode == CONTACT_UNIT_CONSERVATIVE:
                st.info(_t(
                    st,
                    "GED 后不进行配体接触单元分组，保留原子级相互作用标签用于频率分析和候选筛选。",
                    "After GED, atom-level interaction labels are retained for frequency analysis and candidate screening.",
                ))
            else:
                st.caption(_t(
                    st,
                    "已加载项目使用隐藏的旧版兼容策略；保存时保持原值。",
                    "The loaded project uses a hidden legacy policy; saving preserves it unchanged.",
                ))

            # Advanced review and grouping controls remain available in YAML
            # for reproducibility but are not exposed in the compact GUI.
            signature_policy = str(values.get("class_signature_policy", "most_common"))
            split_classes = bool(values["split_nonidentical_classes"])
            manual_units = str(values["manual_ligand_units_yaml"])
            recommendation_fraction = float(values["recommendation_consensus_min_fraction"])
            recommendation_smiles = str(values["recommendation_ligand_smiles_yaml"])
            type_map = str(values["interaction_type_map_yaml"])
            st.caption(_t(
                st,
                "相互作用标签统一分为 Polar、Hydrophobic、π 和 Metal contacts。",
                "Interaction labels are grouped as Polar, Hydrophobic, π, and Metal contacts.",
            ))

        with st.expander("Step09 · " + _t(st, "基本输出格式", "basic output formats")):
            formats = st.multiselect(_t(st, "输出格式", "Output formats"), ["svg", "png"], default=list(values["plot_formats"]))
            width = float(values["plot_width"])
            height = float(values["plot_height"])
            dpi = int(values["plot_dpi"])
            grid = int(values["plot_grid_cols"])
            spacing = float(values["entity_spacing"])
            scale = float(values["substrate_scale"])
            rotate = float(values["global_rotate"])
            node_size = int(values["plot_node_size"])
            font_size = float(values["plot_font_size"])
            ligand_node_scale = float(values["plot_ligand_node_scale"])
            zoom = float(values["layout_zoom"])
            entity_order = str(values.get("entity_order", ""))
            base_distance = float(values["residue_base_distance"])
            core_distance = float(values["residue_core_min_distance"])
            residue_distance_plot = float(values["residue_residue_min_distance"])
            swap_substrates = bool(values["swap_substrates"])
            substrate_rotations = str(values["rotate_substrates_yaml"])
            entity_offsets = str(values["entity_y_offsets_yaml"])
            residue_offsets = str(values["manual_residue_offsets_yaml"])
            node_offsets = str(values["manual_node_offsets_yaml"])

        with st.expander(_t(st, "可选配体原子选择 / 裁剪", "Optional ligand-atom selection / trimming")):
            st.caption(_t(st, "默认关闭。它会改变进入后续分析的配体原子，因此只有在研究设计明确要求分析反应核心时才启用。", "Disabled by default. It changes which ligand atoms enter downstream analysis; enable it only when the study design explicitly analyzes a reaction-core subset."))
            trim_enabled = st.checkbox(_t(st, "启用配体原子选择", "Enable ligand-atom selection"), bool(values["trim_enabled"]))
            timing_options = ["before_docking", "after_clash"]
            trim_timing = st.selectbox(
                _t(st, "选择时机", "Selection timing"), timing_options,
                index=timing_options.index(str(values.get("trim_timing", "before_docking"))),
                format_func=lambda item: _t(st, "对接前（裁剪模板底物）", "Before docking (template ligand)") if item == "before_docking" else _t(st, "碰撞筛选后（裁剪复合物）", "After clash filtering (complexes)"),
            )
            trim_rules = st.text_area(
                _t(st, "原子选择规则（YAML 列表）", "Atom-selection rules (YAML list)"),
                str(values["trim_rules_yaml"]), height=150, disabled=not trim_enabled,
                help="Example below keeps only named atoms in ATP.",
            )
            st.code("- rule_id: ATP_reaction_core\n  match:\n    resname: ATP\n  action: keep_only\n  keep_atom_names: [P1, P2, P3, O1, O2, O3]", language="yaml")

        preview_submit = st.form_submit_button(
            _t(st, "预览接触单元规则", "Preview contact-unit rules"),
            disabled=contact_unit_mode != CONTACT_UNIT_UNIVERSAL,
        )
        apply_submit = st.form_submit_button(_t(st, "应用全部参数", "Apply all parameters"), type="primary")

    if apply_submit or preview_submit:
        updates = {
            "step02_alignment_mode": alignment, "step02_multi_ligand_mode": multi,
            "step02_input_source": input_source, "min_sequence_identity": identity,
            "min_alignment_coverage": coverage, "min_matched_ca": matched_ca,
            "step02_overwrite": overwrite, "clash_max": clash_max, "clash_delta": clash_delta,
            "step03_include_resnames": include_names, "catalytic_distance_cutoff": cutoff,
            "catalytic_mean_plddt_min": mean_plddt, "catalytic_p10_plddt_min": p10_plddt,
            "plip_executable": plip_exe, "plip_ligand_node_id": node_id,
            "plip_connectivity_source": connectivity, "plip_run": plip_run,
            "plip_save_intermediate": plip_save_intermediate,
            "require_graph_occurrence": require_graph_occurrence,
            "consurf_mapping_enabled": mapping_enabled,
            "consurf_mapping_min_target_coverage": mapping_target,
            "consurf_mapping_min_contact_coverage": mapping_contact,
            "consurf_mapping_fail_on_low_coverage": mapping_fail,
            "consurf_grade_min": consurf,
            "residue_ligand_distance_max": residue_distance, "occurrence_frequency_min": occurrence,
            "ged_threshold": ged_threshold, "ged_max_workers": workers,
            "ged_timeout_seconds": timeout, "ged_require_same_edge_count": same_edges,
            "common_frequency_min": common, "rare_frequency_max": rare,
            "contact_unit_mode": contact_unit_mode,
            "equivalence_mode": values.get("equivalence_mode", "standard"),
            "class_signature_policy": signature_policy,
            "split_nonidentical_classes": split_classes, "manual_ligand_units_yaml": manual_units,
            "recommendation_consensus_min_fraction": recommendation_fraction,
            "recommendation_ligand_smiles_yaml": recommendation_smiles,
            "recommendation_require_acceptance": contact_unit_mode == CONTACT_UNIT_GUIDED,
            "accepted_recommendation_ids_yaml": values["accepted_recommendation_ids_yaml"],
            "rejected_recommendation_ids_yaml": values["rejected_recommendation_ids_yaml"],
            "interaction_type_map_yaml": type_map,
            "auto_detect_ligand_units": contact_unit_mode in {CONTACT_UNIT_UNIVERSAL, CONTACT_UNIT_GUIDED},
            "plot_formats": formats, "plot_width": width, "plot_height": height,
            "plot_dpi": dpi, "plot_grid_cols": grid, "entity_spacing": spacing,
            "substrate_scale": scale, "global_rotate": rotate,
            "plot_node_size": node_size, "plot_font_size": font_size,
            "plot_ligand_node_scale": ligand_node_scale, "layout_zoom": zoom,
            "entity_order": entity_order, "residue_base_distance": base_distance,
            "residue_core_min_distance": core_distance,
            "residue_residue_min_distance": residue_distance_plot,
            "swap_substrates": swap_substrates,
            "entity_y_offsets_yaml": entity_offsets,
            "rotate_substrates_yaml": substrate_rotations,
            "manual_residue_offsets_yaml": residue_offsets,
            "manual_node_offsets_yaml": node_offsets,
            "trim_enabled": trim_enabled,
            "trim_timing": trim_timing, "trim_rules_yaml": trim_rules,
            "extra_yaml": values.get("extra_yaml", "{}"),
        }
        _set_values(st, updates)
        if contact_unit_mode == CONTACT_UNIT_CONSERVATIVE:
            st.session_state.ccg_equivalence_preview = []
            st.session_state.ccg_equivalence_preview_inventory = []
            st.session_state.ccg_equivalence_preview_report = {}
            st.session_state.ccg_equivalence_preview_input_audit = []
        if preview_submit:
            try:
                records, inventory, report, input_audit = _build_recommendation_preview(build_config(_values(st)))
                st.session_state.ccg_equivalence_preview = records
                st.session_state.ccg_equivalence_preview_inventory = inventory
                st.session_state.ccg_equivalence_preview_report = report
                st.session_state.ccg_equivalence_preview_input_audit = input_audit
            except Exception as exc:
                st.session_state.ccg_equivalence_preview = []
                st.session_state.ccg_equivalence_preview_inventory = []
                st.session_state.ccg_equivalence_preview_report = {"generation_error": str(exc)}
                st.error(_t(st, f"无法生成建议：{exc}", f"Could not generate proposals: {exc}"))
        if apply_submit:
            st.rerun()
    _render_equivalence_recommendation_preview(st)
    config, preview_yaml = _config_preview(_values(st))
    if config is not None:
        st.download_button(_t(st, "下载当前完整 YAML", "Download current complete YAML"), preview_yaml, f"{_values(st)['target_id']}_af2_catgraph.yaml", "text/yaml")


def _render_equivalence_recommendation_preview(st: Any) -> None:
    records = st.session_state.get("ccg_equivalence_preview", [])
    inventory = st.session_state.get("ccg_equivalence_preview_inventory", [])
    report = st.session_state.get("ccg_equivalence_preview_report", {})
    input_audit = st.session_state.get("ccg_equivalence_preview_input_audit", [])
    if not records and not inventory and not report and not input_audit:
        return
    st.markdown("#### " + _t(st, "配体接触单元建议审查", "Ligand contact-unit proposal review"))
    import pandas as pd
    input_issues = [
        row for row in input_audit
        if not row.get("exists") or row.get("status") not in {"ready", "actual_step08_recommendations"}
    ]
    if input_issues:
        with st.expander(_t(st, "输入检查", "Input check")):
            st.dataframe(pd.DataFrame(input_issues), use_container_width=True, hide_index=True)
    if report.get("generation_error"):
        st.error(str(report["generation_error"]))
        return
    if report.get("preview_source") == "completed_step08":
        st.info(_t(
            st,
            "当前显示的是上一次 Step08 根据完整 PLIP 构象集合生成的正式建议。接受并保存后，请重新运行 Step08。",
            "These are the formal proposals generated from the complete PLIP ensemble in the previous Step08 run. Accept and save them, then rerun Step08.",
        ))
    elif report.get("preview_source") == "template_ligand_preliminary":
        st.warning(_t(
            st,
            "这是仅根据模板底物生成的初步建议。完成一次 Step08 后，请重新预览并优先审查完整构象集合生成的正式建议。",
            "This is a preliminary template-ligand preview. After one Step08 run, preview again and review the formal ensemble-derived proposals.",
        ))
    if inventory:
        with st.expander(_t(st, "检测到的配体原子", "Detected ligand atoms")):
            st.dataframe(pd.DataFrame(inventory), use_container_width=True, hide_index=True)
    if not records:
        st.info(_t(st, "没有生成可供审查的接触单元建议；配体原子会保持独立标签。", "No contact-unit proposal was generated for review; ligand atoms remain independently labeled."))
        return
    table = pd.DataFrame(records)
    visible = [column for column in ["recommendation_id", "status", "resname", "proposed_unit", "atoms", "connectivity_evidence", "enzyme_contact_assessment", "confidence", "rationale"] if column in table]
    st.dataframe(table[visible], use_container_width=True, hide_index=True)
    values = _values(st)
    if str(values.get("contact_unit_mode")) == CONTACT_UNIT_UNIVERSAL:
        applied = sum(str(row.get("status")) == "applied" for row in records)
        st.success(_t(
            st,
            f"通用规则模式已识别并将自动应用 {applied} 个接触单元；无需逐项接受。",
            f"Universal mode identified {applied} contact units that will be applied automatically; no per-rule acceptance is required.",
        ))
        return
    selectable = [str(row["recommendation_id"]) for row in records if row.get("recommendation_id")]
    try:
        current_accepted = as_yaml_list(str(values.get("accepted_recommendation_ids_yaml", "[]")), "accepted IDs")
        current_rejected = as_yaml_list(str(values.get("rejected_recommendation_ids_yaml", "[]")), "rejected IDs")
    except ValueError:
        current_accepted, current_rejected = [], []
    accepted = st.multiselect(_t(st, "接受", "Accept"), selectable, default=[item for item in current_accepted if item in selectable], key="ccg_recommendation_accepted_select")
    rejected_options = [item for item in selectable if item not in accepted]
    rejected = st.multiselect(_t(st, "拒绝", "Reject"), rejected_options, default=[item for item in current_rejected if item in rejected_options], key="ccg_recommendation_rejected_select")
    if st.button(_t(st, "保存审查决定", "Save review decisions"), key="ccg_save_recommendation_decisions"):
        _set_values(st, {
            "accepted_recommendation_ids_yaml": yaml.safe_dump(accepted, sort_keys=False),
            "rejected_recommendation_ids_yaml": yaml.safe_dump(rejected, sort_keys=False),
        })
        try:
            path = write_config(st.session_state.ccg_config_path, build_config(_values(st)))
            st.session_state.ccg_config_path = str(path)
            saved = load_yaml_config(path)
            saved_recommendations = (((saved.get("step09", {}) or {}).get("canonicalization", {}) or {}).get("recommendations", {}) or {})
            saved_accepted = [str(item) for item in saved_recommendations.get("accepted_recommendation_ids", []) or []]
            if sorted(saved_accepted) != sorted(str(item) for item in accepted):
                raise ValueError("The saved YAML did not preserve the selected accepted recommendation IDs.")
            st.success(_t(st, f"审查决定已写入并验证：{path}。请重新运行 Step08 才会应用合并。", f"Review decisions were written and verified in: {path}. Rerun Step08 to apply the merge."))
        except (OSError, ValueError) as exc:
            st.error(_t(st, f"无法保存审查决定：{exc}", f"Could not save review decisions: {exc}"))


def _job_running(st: Any) -> bool:
    job = st.session_state.get("ccg_job")
    return bool(job and job.running)


def _start_job(st: Any, command: list[str]) -> None:
    if _job_running(st):
        st.error(_t(st, "已有任务正在运行。", "A task is already running.")); return
    st.session_state.ccg_job = BackgroundJob(command=command, cwd=REPOSITORY_ROOT)


def _run_command(config_path: str, choice: str, max_structures: int, clean: bool, values: dict[str, Any] | None = None) -> list[str]:
    base = [sys.executable, "-m", "catcongraph.cli"]
    if choice == "preflight":
        command = base + ["preflight", "--config", config_path]
        if max_structures:
            command += ["--max-structures", str(max_structures)]
    elif choice == "all":
        command = base + ["run", "--config", config_path, "--stage", "all"]
    else:
        command = base + [choice, "--config", config_path]
        if choice in {"step02", "step05", "step06", "step07"} and max_structures:
            command += ["--max-structures", str(max_structures)]
    if clean and choice not in {"preflight", "step01", "step03", "step04", "step10"}:
        command.append("--clean")
    values = values or {}
    return _conda_wrapped_command(
        command, values.get("conda_executable", "conda"), values.get("workflow_conda_env", ""),
    )


def _render_job_panel(st: Any) -> None:
    job = st.session_state.get("ccg_job")
    if not job:
        return
    job.refresh()
    if job.running:
        st.info(_t(st, "任务在后台运行；点击刷新读取新日志。关闭浏览器不会主动停止服务器进程。", "Task is running in the background; refresh reads new logs. Closing the browser does not stop the server process."))
        a, b, _ = st.columns([1, 1, 4])
        if a.button(_t(st, "刷新状态", "Refresh status")):
            st.rerun()
        if b.button(_t(st, "停止任务", "Stop task")):
            job.stop()
    elif job.returncode == 0:
        st.success(_t(st, "任务完成（退出代码 0）。", "Task completed (exit code 0)."))
    else:
        st.error(_t(st, f"任务结束（退出代码 {job.returncode}）。请查看日志。", f"Task ended (exit code {job.returncode}). Check the log."))
    with st.expander(_t(st, "详细命令与运行日志", "Detailed command and run log"), expanded=not job.running and job.returncode not in (None, 0)):
        st.code(" ".join(job.command), language="bash")
        st.text_area(_t(st, "运行日志（最新 1000 行）", "Run log (latest 1000 lines)"), "\n".join(job.lines[-1000:]) or "…", height=340, disabled=True)


def _render_run(st: Any) -> None:
    st.subheader(_t(st, "运行与监控", "Run and monitor"))
    values = _values(st)
    with st.form("workflow_environment_form", clear_on_submit=False):
        a, b = st.columns(2)
        conda_executable = a.text_input(_t(st, "Conda 可执行程序", "Conda executable"), str(values.get("conda_executable", "conda")))
        workflow_env = b.text_input(_t(st, "后续工作流 Conda 环境", "Workflow Conda environment"), str(values.get("workflow_conda_env", "")), help=_t(st, "留空表示使用启动 AF2-CatGraph 的当前环境。", "Leave blank to use the environment that launched AF2-CatGraph."))
        apply_environment = st.form_submit_button(_t(st, "应用工作流环境", "Apply workflow environment"))
    if apply_environment:
        _set_values(st, {"conda_executable": conda_executable, "workflow_conda_env": workflow_env})
        st.rerun()
    active_environment = str(values.get("workflow_conda_env", "")).strip() or _t(st, "当前 AF2-CatGraph 环境", "current AF2-CatGraph environment")
    st.info(_t(st, f"Step00–09 将在以下环境运行：{active_environment}", f"Step00–09 will run in: {active_environment}"))
    path_text = st.text_input(_t(st, "要运行的 YAML", "YAML to run"), st.session_state.ccg_config_path, key="ccg_run_config")
    st.session_state.ccg_config_path = path_text
    path = Path(path_text).expanduser()
    if path.exists(): st.success(_t(st, "已找到配置文件。", "Configuration file found."))
    else: st.warning(_t(st, "未找到配置文件。请先在项目设置中保存。", "Configuration file was not found. Save it in Project setup first."))
    labels = _step_labels(st)
    choice = st.selectbox(_t(st, "运行内容", "Run"), STEP_KEYS, format_func=labels.get, key="ccg_run_choice")
    if choice == "step10" and not bool(values.get("trim_enabled", False)):
        st.warning(_t(
            st,
            "Step10 当前未启用。请先在“步骤参数与配体接触单元分组 → 可选配体原子选择 / 裁剪”中启用并保存 YAML。",
            "Step10 is currently disabled. Enable it under Step parameters & contact units → Optional ligand-atom selection, then save the YAML.",
        ))
    if path.exists() and choice in {"step08", "all"}:
        try:
            on_disk = load_yaml_config(path)
            recommendations = (
                (((on_disk.get("step09", {}) or {}).get("canonicalization", {}) or {})
                 .get("recommendations", {}) or {})
            )
            canonicalization = ((on_disk.get("step09", {}) or {}).get("canonicalization", {}) or {})
            saved_contact_mode = str(canonicalization.get("contact_unit_mode", CONTACT_UNIT_GUIDED))
            accepted_on_disk = recommendations.get("accepted_recommendation_ids", []) or []
            if saved_contact_mode == CONTACT_UNIT_CONSERVATIVE:
                st.info(_t(
                    st,
                    "当前配置使用保守人工定义策略：不会加载预置规则或建议；留空人工接触单元时保持原子级分类。",
                    "This config uses conservative manual definition: no presets or proposals are loaded; atom-level classification is retained when manual contact units are empty.",
                ))
            elif saved_contact_mode == CONTACT_UNIT_UNIVERSAL:
                st.info(_t(
                    st,
                    "当前配置会在 GED 后自动应用 S4 通用配体接触单元规则，再进行频率统计和候选构象筛选。",
                    "This config automatically applies the universal S4 ligand contact-unit rules after GED, then performs frequency analysis and candidate screening.",
                ))
            elif accepted_on_disk:
                st.success(_t(st, f"运行配置中已保存 {len(accepted_on_disk)} 个接触单元接受项。", f"The run configuration contains {len(accepted_on_disk)} saved contact-unit acceptance decision(s)."))
            else:
                st.info(_t(st, "运行配置中没有已接受的新增接触单元建议。默认建议和人工填写规则仍会按配置应用。", "The run configuration contains no accepted additional contact-unit suggestion. Default suggestions and manual rules are still applied as configured."))
        except Exception as exc:
            st.warning(_t(st, f"无法检查接触单元策略：{exc}", f"Could not inspect the contact-unit policy: {exc}"))
    a, b, c = st.columns([1, 1, 2])
    maximum = a.number_input(_t(st, "最多处理结构数（0=全部）", "Maximum structures (0=all)"), 0, 1000000, 3)
    clean = b.checkbox(_t(st, "清理该步骤旧输出", "Clean prior output for this step"), False)
    command = _run_command(path_text, choice, int(maximum), clean, values)
    with c.expander(_t(st, "查看命令", "Show command")):
        st.code(" ".join(command), language="bash")
    if st.button(_t(st, "启动所选任务", "Start selected task"), type="primary", disabled=not path.exists() or _job_running(st)):
        _start_job(st, command); st.rerun()
    st.divider(); _render_job_panel(st)


def _resolve_output_root(values: dict[str, Any]) -> Path:
    root = Path(str(values.get("output_root", "results/MY_ENZYME"))).expanduser()
    return root if root.is_absolute() else REPOSITORY_ROOT / root


def _cached_files(st: Any, root: Path, include_intermediates: bool) -> list[Path]:
    @st.cache_data(ttl=4, show_spinner=False)
    def scan(root_text: str, include: bool) -> list[str]:
        return [str(path) for path in result_files(root_text, include)]
    return [Path(path) for path in scan(str(root), include_intermediates)]


def _file_category(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".xlsx", ".json", ".txt", ".md", ".html", ".yaml", ".yml"}: return "table"
    if suffix in {".png", ".svg", ".pdf", ".jpg", ".jpeg"}: return "figure"
    if suffix in {".pdb", ".cif", ".mmcif"}: return "structure"
    return "other"


def _preview_result_file(st: Any, path: Path) -> None:
    suffix = path.suffix.lower()
    try:
        size = path.stat().st_size
        binary = path.read_bytes()
    except OSError as exc:
        st.error(_t(st, f"无法读取结果文件：{exc}", f"Cannot read result file: {exc}"))
        return
    mime = {".csv":"text/csv", ".tsv":"text/tab-separated-values", ".json":"application/json", ".xlsx":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".png":"image/png", ".svg":"image/svg+xml", ".pdf":"application/pdf", ".pdb":"chemical/x-pdb", ".txt":"text/plain"}.get(suffix, "application/octet-stream")
    st.download_button(_t(st, "下载此文件", "Download this file"), binary, path.name, mime)
    st.caption(f"{path.name} · {size / 1024:.1f} KiB")
    with st.expander(_t(st, "文件位置", "File location")):
        st.code(str(path), language="text")
    if size > 25 * 1024 * 1024:
        st.warning(_t(st, "文件大于 25 MiB；为避免浏览器卡顿，仅提供下载。", "File exceeds 25 MiB; download is provided to avoid browser lag.")); return
    try:
        if suffix in {".csv", ".tsv"}:
            import pandas as pd
            table = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
            st.dataframe(table.head(2000), use_container_width=True, height=480)
        elif suffix == ".xlsx":
            import pandas as pd
            book = pd.ExcelFile(path); sheet = st.selectbox(_t(st, "工作表", "Worksheet"), book.sheet_names, key=f"sheet_{path}")
            st.dataframe(pd.read_excel(path, sheet_name=sheet).head(2000), use_container_width=True, height=480)
        elif suffix in {".json", ".yaml", ".yml"}:
            st.json(yaml.safe_load(binary.decode("utf-8", errors="replace")))
        elif suffix in {".png", ".jpg", ".jpeg", ".svg"}: st.image(binary, use_container_width=True)
        elif suffix == ".html":
            st.warning(_t(st, "HTML 结果可能包含脚本。仅在信任该文件来源时启用交互预览。", "HTML results may contain scripts. Enable interactive preview only when you trust the file source."))
            allow_html = st.checkbox(_t(st, "我信任此文件，启用 HTML 预览", "I trust this file; enable HTML preview"), False, key=f"allow_html_{path}")
            if not allow_html:
                return
            import streamlit.components.v1 as components
            components.html(binary.decode("utf-8", errors="replace"), height=720, scrolling=True)
        elif suffix == ".pdf":
            # ``st.pdf`` is not available in the oldest supported Streamlit
            # versions, so use a small self-contained browser embed instead.
            import streamlit.components.v1 as components
            encoded = base64.b64encode(binary).decode("ascii")
            components.html(f'<embed src="data:application/pdf;base64,{encoded}" width="100%" height="700px" type="application/pdf">', height=720)
        else:
            text = binary.decode("utf-8", errors="replace")
            if suffix in {".pdb", ".cif", ".mmcif"}:
                st.info(_t(st, f"结构文本预览：{sum(x.startswith(('ATOM  ', 'HETATM')) for x in text.splitlines()):,} 条原子记录。", f"Structure preview: {sum(x.startswith(('ATOM  ', 'HETATM')) for x in text.splitlines()):,} atom records."))
            st.code(text[:50000], language="text")
    except Exception as exc:
        st.error(_t(st, f"无法预览文件：{exc}", f"Cannot preview this file: {exc}"))


def _row_label(row: dict[str, str]) -> str:
    return str(row.get("label") or row.get("merged_class") or row.get("original_class") or "class")


def _render_selected_class_viewer(st: Any, root: Path, values: dict[str, Any], chosen: dict[str, str]) -> None:
    """Render the selected 3D viewer before the class-card grid.

    Keeping the viewer above the grid makes the effect of a button click
    immediately visible.  The old layout created the viewer only after every
    class card, which could look like a non-responsive button on long grids.
    """
    selected = _row_label(chosen)
    st.subheader(_t(st, f"{selected} · 活性口袋三维查看", f"{selected} · active-pocket 3D viewer"))
    pdb_path = representative_pdb(root, chosen, REPOSITORY_ROOT)
    if not pdb_path:
        st.error(_t(st, "找不到该类代表结构 PDB。请检查步骤 04/09 输出是否仍在原路径。", "Representative PDB was not found. Check whether the Step04/09 outputs remain at their original path.")); return
    pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
    interactions = class_interactions(chosen, root, REPOSITORY_ROOT)
    try:
        page, status = viewer_html(pdb_text, interactions, selected)
    except Exception as exc:
        st.error(_t(st, f"三维视图无法生成：{exc}", f"3D view could not be generated: {exc}")); return
    st.caption(_t(st, "蛋白为 cartoon；底物为 sticks/spheres；接触残基为红色 sticks；虚线依据 Step05 PLIP 的残基–配体记录重建。", "Protein is cartoon; ligands are sticks/spheres; contacting residues are red sticks; dashed lines are reconstructed from Step05 PLIP residue–ligand records."))
    if page:
        import streamlit.components.v1 as components
        components.html(page, height=680, scrolling=False)
        st.success(status)
    else: st.warning(status)
    a, b = st.columns(2)
    a.download_button(_t(st, "下载代表性 PDB", "Download representative PDB"), pdb_text, pdb_path.name, "chemical/x-pdb")
    script = pymol_script(pdb_path.name, interactions, str(values.get("ligand_resnames", "LIG")).replace(",", " ").split())
    b.download_button(_t(st, "下载 PyMOL 脚本（.pml）", "Download PyMOL script (.pml)"), script, f"{selected}_pocket_view.pml", "text/plain")


def _render_class_viewer(st: Any, root: Path, values: dict[str, Any]) -> None:
    rows = class_rows(root)
    if not rows:
        st.info(_t(st, "尚未找到 Step09 类别汇总表。先完成 Step09，再回到这里。", "No Step09 class summary was found. Complete Step09 and return here.")); return
    st.caption(_t(st, "点击类别卡片下的按钮后，三维结构会立即显示在此处；该浏览器使用内置 3Dmol.js，不需要 PyMOL。", "After clicking a class card, its 3D structure appears here immediately. This browser uses built-in 3Dmol.js and does not require PyMOL."))

    selected = st.session_state.get("ccg_selected_class", "")
    chosen = next((row for row in rows if _row_label(row) == selected), None)
    if chosen:
        with st.container(border=True):
            _render_selected_class_viewer(st, root, values, chosen)
        st.divider()

    for start in range(0, len(rows), 3):
        for col, row in zip(st.columns(3), rows[start:start + 3]):
            label = _row_label(row)
            panel = Path(str(row.get("panel_png", "")))
            if panel.exists(): col.image(panel.read_bytes(), use_container_width=True)
            else: col.caption(_t(st, "未找到 PNG 类别图。", "PNG class plot not found."))
            col.caption(f"{label} · n={row.get('n_structures', '?')}")
            if col.button(_t(st, f"查看 {label} 的三维结构", f"View {label} in 3D"), key=f"class_{label}"):
                st.session_state.ccg_selected_class = label
                st.rerun()


def _render_results(st: Any) -> None:
    values, root = _values(st), _resolve_output_root(_values(st))
    st.subheader(_t(st, "结果与可视化浏览器", "Results and visualization browser"))
    st.caption(f"{_t(st, '当前项目', 'Current project')}: `{values.get('target_id', 'MY_ENZYME')}`")
    with st.expander(_t(st, "结果目录位置", "Results directory location")):
        st.code(str(root), language="text")
    if not root.exists():
        st.info(_t(st, "结果目录尚不存在。完成至少一个步骤后再刷新。", "The results directory does not yet exist. Refresh after completing at least one step.")); return
    summaries = class_rows(root)
    structure_total = 0
    for row in summaries:
        try:
            structure_total += int(float(str(row.get("n_structures", "0") or "0")))
        except ValueError:
            continue
    a, b, c = st.columns(3)
    a.metric(_t(st, "最终类别", "Final classes"), len(summaries) if summaries else "—")
    b.metric(_t(st, "类别内结构", "Structures in classes"), structure_total if summaries else "—")
    c.metric(_t(st, "已完成主流程步骤", "Completed main steps"), f"{sum(done for _, done in _workflow_progress(values))}/9")
    classes_tab, files_tab = st.tabs([_t(st, "类别三维查看", "Class 3D viewer"), _t(st, "所有结果文件", "All result files")])
    with classes_tab: _render_class_viewer(st, root, values)
    with files_tab:
        include = st.checkbox(_t(st, "包括大量中间文件（Step05/PLIP；扫描会更慢）", "Include large intermediate files (Step05/PLIP; scan is slower)"), False)
        if st.button(_t(st, "刷新文件列表", "Refresh file list")):
            st.cache_data.clear(); st.rerun()
        files = _cached_files(st, root, include)
        counts = {kind: sum(_file_category(path) == kind for path in files) for kind in ("table", "figure", "structure", "other")}
        for col, key, zh, en in zip(st.columns(4), ("table", "figure", "structure", "other"), ("表格/文本", "图形", "结构", "其他"), ("Tables/text", "Figures", "Structures", "Other")):
            col.metric(_t(st, zh, en), counts[key])
        category = st.radio(_t(st, "文件类别", "File category"), ["all", "table", "figure", "structure", "other"], horizontal=True, format_func=lambda item: {"all":_t(st,"全部","All"), "table":_t(st,"表格/文本","Tables/text"), "figure":_t(st,"图形","Figures"), "structure":_t(st,"结构","Structures"), "other":_t(st,"其他","Other")}[item])
        filtered = files if category == "all" else [path for path in files if _file_category(path) == category]
        if not filtered: st.info(_t(st, "没有匹配的文件。", "No matching files.")); return
        labels = [str(path.relative_to(root)) for path in filtered]
        selected_label = st.selectbox(_t(st, "选择结果文件", "Choose a result file"), labels)
        _preview_result_file(st, filtered[labels.index(selected_label)])



def _render_manual(st: Any) -> None:
    st.subheader(_t(st, "使用手册", "Manual"))
    st.markdown(_t(st, """
AF2-CatGraph 的主工作流为 **00–09**：00 检查输入与依赖；01 全局 pLDDT；02 模板引导放置；03 碰撞筛选；04 催化区域 pLDDT；05 PLIP；06 关键残基；07 GED 分类；08 配体接触单元分组；09 汇总与可视化。可选 Step10 用于底物/配体原子选择与剪切，不包含在“运行全部主流程”中。

建议先保存 YAML 并运行 00，再用 3–5 个结构试运行 01–09。每一步都保留 CSV/JSON 审计结果；常用科学参数可在“步骤参数与配体接触单元分组”中调整。

结果浏览器可以预览表格、图形和结构，并下载原始文件。旧版本生成的文件名仍可读取。
""", """
The main AF2-CatGraph workflow is **00–09**: 00 input/dependency checks; 01 global pLDDT; 02 template-guided placement; 03 clash filtering; 04 catalytic-region pLDDT; 05 PLIP; 06 key residues; 07 GED classification; 08 ligand contact-unit grouping; and 09 summary/visualization. Optional Step10 performs substrate/ligand atom selection and trimming and is not included in Run all.

Save the YAML and run 00 first, then pilot 01–09 on 3–5 structures. Every step keeps auditable CSV/JSON outputs; commonly adjusted scientific settings are available under Step parameters and ligand contact-unit grouping.

The Results browser previews tables, figures, and structures and downloads the original files. Legacy artifact names remain readable.
"""))
    for path, zh, en in [
        (REPOSITORY_ROOT / "docs" / "GUI_GUIDE.md", "下载完整使用手册", "Download full user guide"),
        (REPOSITORY_ROOT / "docs" / "AF2_PREPROCESS.md", "下载 AF2 预处理说明", "Download AF2 preprocessing guide"),
        (REPOSITORY_ROOT / "docs" / "CONFIG_REFERENCE.md", "下载配置参考", "Download configuration reference"),
        (REPOSITORY_ROOT / "docs" / "CONTACT_UNIT_REVIEW.md", "下载配体接触单元分组指南", "Download ligand contact-unit grouping guide"),
    ]:
        if path.exists():
            st.download_button(_t(st, zh, en), path.read_bytes(), path.name, "text/markdown")


def main() -> None:
    st = _st()
    identity = build_identity(REPOSITORY_ROOT, __version__)
    refresh_running_build(REPOSITORY_ROOT, identity)
    st.set_page_config(page_title="AF2-CatGraph", page_icon="🧬", layout="wide")
    _initialize_state(st); _show_css(st)
    nav = [("overview", "总览", "Overview"), ("colabfold", "步骤 0 · ColabFold", "Step 0 · ColabFold"), ("af2_preprocess", "步骤 0A · AF2 预处理", "Step 0A · AF2 preprocessing"), ("project", "项目设置", "Project setup"), ("advanced", "步骤参数与配体接触单元分组", "Step parameters & contact-unit grouping"), ("run", "运行与监控", "Run and monitor"), ("results", "结果浏览", "Results browser"), ("manual", "使用手册", "Manual")]
    with st.sidebar:
        st.title("AF2-CatGraph")
        st.selectbox("Language / 语言", ["English", "中文"], key="ccg_language")
        labels = {key: _t(st, zh, en) for key, zh, en in nav}
        section = st.radio(_t(st, "导航", "Navigation"), [key for key, _, _ in nav], format_func=labels.get, label_visibility="collapsed")
        st.divider(); st.caption(_t(st, "当前项目", "Current project")); st.code(str(_values(st).get("target_id", "MY_ENZYME")))
        if st.button(_t(st, "重置当前界面状态", "Reset current UI state"), use_container_width=True):
            for key in list(st.session_state):
                if str(key).startswith("ccg_"):
                    del st.session_state[key]
            st.rerun()
        with st.expander(_t(st, "帮助与诊断", "Help and diagnostics"), expanded=False):
            st.caption(_t(
                st,
                "此哈希由当前正在执行的源文件计算。提交新版本后它必须改变；若不变，说明服务器仍在运行旧代码而不是浏览器缓存问题。",
                "This hash is calculated from the source currently executing. It must change after a code deployment; if it does not, the server is still running old code rather than the browser using a cache.",
            ))
            st.code(f"source root: {identity.source_root}\nsource hash: {identity.source_fingerprint}\nrevision: {identity.revision or 'archive/no git metadata'}")
    renderers: dict[str, Callable[[Any], None]] = {"overview":_render_overview, "colabfold":_render_colabfold, "af2_preprocess":_render_af2_preprocess, "project":_render_project_setup, "advanced":_render_advanced, "run":_render_run, "results":_render_results, "manual":_render_manual}
    renderers[section](st)


if __name__ == "__main__":
    main()
