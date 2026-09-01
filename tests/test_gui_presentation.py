from pathlib import Path
import inspect
from types import SimpleNamespace

from catcongraph import gui


def test_project_readiness_counts_only_existing_inputs(tmp_path: Path):
    af2 = tmp_path / "af2"
    af2.mkdir()
    template = tmp_path / "template.pdb"
    template.write_text("END\n", encoding="utf-8")
    ligand = tmp_path / "ligand.pdb"
    ligand.write_text("END\n", encoding="utf-8")

    count, checks = gui._project_readiness({
        "af2_dir": str(af2),
        "template_protein_pdb": str(template),
        "ligand_paths": str(ligand),
        "conservation_file": str(tmp_path / "missing.zip"),
        "ged_executable": str(tmp_path / "missing-ged"),
    })

    assert count == 3
    assert [ready for _, ready in checks] == [True, True, True, False, False]


def test_workflow_progress_is_presentation_only_directory_check(tmp_path: Path):
    (tmp_path / "01_af2_prepare").mkdir()
    (tmp_path / "04_catalytic_region_plddt_qc").mkdir()

    progress = gui._workflow_progress({"output_root": str(tmp_path)})

    assert len(progress) == 9
    assert progress[0] == ("01", True)
    assert progress[3] == ("04", True)
    assert sum(done for _, done in progress) == 2


def test_named_conda_environments_wrap_commands_without_shell_activation() -> None:
    wrapped = gui._conda_wrapped_command(
        ["colabfold_batch", "input.fasta", "output"], "conda", "colabfold-env",
    )
    assert wrapped == [
        "conda", "run", "-n", "colabfold-env", "--no-capture-output",
        "colabfold_batch", "input.fasta", "output",
    ]


def test_workflow_command_uses_the_workflow_environment() -> None:
    command = gui._run_command(
        "project.yaml", "step05", 3, False,
        {"conda_executable": "conda", "workflow_conda_env": "catcongraph-env"},
    )
    assert command[:6] == [
        "conda", "run", "-n", "catcongraph-env", "--no-capture-output", "python",
    ]
    assert command[-2:] == ["--max-structures", "3"]


def test_optional_step10_is_visible_and_does_not_receive_clean_flag() -> None:
    assert "step10" in gui.STEP_KEYS
    fake_st = SimpleNamespace(session_state={"ccg_language": "English"})
    assert "Ligand contact-unit grouping" in gui._step_labels(fake_st)["step08"]


def test_compact_advanced_form_hides_compatibility_controls() -> None:
    source = inspect.getsource(gui._render_advanced)

    assert 'alignment_options = ["auto", "sequence_aware"]' in source
    assert "Coordinate-matching method" not in source
    assert "Chemistry-guided review" not in source
    assert "Manual ligand contact units" not in source
    assert "Grid columns" not in source
    assert "Optional ligand-atom selection" in source

    command = gui._run_command(
        "project.yaml", "step10", 0, True,
        {"conda_executable": "", "workflow_conda_env": ""},
    )

    assert command[-4:] == ["catcongraph.cli", "step10", "--config", "project.yaml"]
    assert "--clean" not in command


def test_gui_defaults_to_english_and_keeps_headings_unclipped() -> None:
    state_source = inspect.getsource(gui._initialize_state)
    css_source = inspect.getsource(gui._show_css)

    assert '"ccg_language", "English"' in state_source
    assert "padding-top:4.15rem!important" in css_source
    assert 'data-testid="stHeadingWithActionElements"' in css_source
    assert "overflow:visible!important" in css_source
    assert "word-break:normal!important" in css_source
    assert "hyphens:none!important" in css_source


def test_supported_target_preview_ignores_a_stale_result_table(tmp_path: Path) -> None:
    output_root = tmp_path / "results" / "AqTMK"
    tables = output_root / "08_chemical_equivalence" / "tables"
    tables.mkdir(parents=True)
    (tables / "AqTMK_step08_ligand_unit_recommendations.csv").write_text(
        "recommendation_id,status,resname,proposed_unit,atoms\n"
        "OLD_GENERIC_RULE,accepted,LIG,old unit,O1;O2\n",
        encoding="utf-8",
    )
    config = {
        "project": {"target_id": "AqTMK", "output_root": str(output_root)},
        "inputs": {},
        "step09": {
            "output_subdir": "08_chemical_equivalence",
            "canonicalization": {
                "contact_unit_mode": "guided_review",
                "recommendations": {"enabled": True, "require_acceptance": True},
            },
        },
    }

    records, _inventory, report, _audit = gui._build_recommendation_preview(config)

    assert len(records) == 7
    assert {record["status"] for record in records} == {"recommended"}
    assert report["preview_source"] == "configured_suggestions"
