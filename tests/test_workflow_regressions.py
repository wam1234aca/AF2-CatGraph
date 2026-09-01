from pathlib import Path

from catcongraph.gui_config import global_rename_config_path, global_rename_values
from catcongraph.gui import _clear_project_widget_state, _project_widget_key
from catcongraph.gui_results import find_class_summary
from catcongraph.io.paths import list_files


def test_af2_rank_files_use_natural_numeric_order(tmp_path: Path) -> None:
    for name in ["GAC.AF2rank100.pdb", "GAC.AF2rank10.pdb", "GAC.AF2rank2.pdb", "GAC.AF2rank1.pdb"]:
        (tmp_path / name).write_text("END\n", encoding="utf-8")

    assert [path.name for path in list_files(tmp_path, ["*.pdb"])] == [
        "GAC.AF2rank1.pdb", "GAC.AF2rank2.pdb", "GAC.AF2rank10.pdb", "GAC.AF2rank100.pdb",
    ]


def test_global_rename_updates_only_target_derived_values() -> None:
    values = {
        "target_id": "GAC",
        "output_root": "results/GAC",
        "af2_dir": "data/GAC/01_af2",
        "ligand_paths": "data/GAC/GAC_ligand.pdb\ndata/shared/MG.pdb",
        "ged_executable": "/tools/GACED/ged",
    }

    renamed = global_rename_values(values, "NEW41")

    assert renamed["target_id"] == "NEW41"
    assert renamed["output_root"] == "results/NEW41"
    assert renamed["af2_dir"] == "data/NEW41/01_af2"
    assert "data/NEW41/NEW41_ligand.pdb" in renamed["ligand_paths"]
    assert renamed["ged_executable"] == "/tools/GACED/ged"


def test_global_rename_updates_gui_yaml_path(tmp_path: Path) -> None:
    renamed = global_rename_config_path(
        "/work/AF2-CatGraph/projects/MY_ENZYME/config.gui.yaml",
        "MY_ENZYME",
        "NewEnzyme",
        tmp_path,
    )
    assert str(renamed).replace("\\", "/") == (
        "/work/AF2-CatGraph/projects/NewEnzyme/config.gui.yaml"
    )

    stale_default = global_rename_config_path(
        "/work/AF2-CatGraph/projects/MY_ENZYME/config.gui.yaml",
        "AqTMK",
        "NewEnzyme",
        tmp_path,
    )
    assert stale_default == (
        tmp_path / "projects" / "NewEnzyme" / "config.gui.yaml"
    )


def test_project_widget_generation_forces_immediate_rename_refresh() -> None:
    class SessionState(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    class FakeStreamlit:
        session_state = SessionState(
            ccg_project_widget_generation=2,
            ccg_field_target_id__2="GAC",
            ccg_field_output_root__2="results/GAC",
        )

    st = FakeStreamlit()
    assert _project_widget_key(st, "ccg_field_target_id") == "ccg_field_target_id__2"

    _clear_project_widget_state(st)

    assert _project_widget_key(st, "ccg_field_target_id") == "ccg_field_target_id__3"
    assert "ccg_field_target_id__2" not in st.session_state
    assert "ccg_field_output_root__2" not in st.session_state


def test_result_summary_prefers_public_step09_name_with_legacy_fallback(tmp_path: Path) -> None:
    tables = tmp_path / "09_class_visualization" / "tables"
    tables.mkdir(parents=True)
    legacy = tables / "GAC_step10_class_visual_summary.csv"
    legacy.write_text("class,n_structures\n1,2\n", encoding="utf-8")
    assert find_class_summary(tmp_path) == legacy

    public = tables / "GAC_step09_class_visual_summary.csv"
    public.write_text("class,n_structures\n1,2\n", encoding="utf-8")
    assert find_class_summary(tmp_path) == public
