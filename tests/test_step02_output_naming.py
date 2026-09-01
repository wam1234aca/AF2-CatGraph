from pathlib import Path

from catcongraph.steps.step02_template_guided_docking import _make_step02_output_name


def test_af2_preprocessed_rank_is_preserved_in_docked_complex_name() -> None:
    output_name, structure_id, pdb_code, rank = _make_step02_output_name(
        Path("NEW2.AF2rank28.pdb"),
        input_order=1,
        target_id="NEW2",
        step02_cfg={},
    )

    assert output_name == "NEW2_28_complex.pdb"
    assert structure_id == "NEW2_28"
    assert pdb_code == "NEW2"
    assert rank == "28"


def test_nonstandard_input_still_falls_back_to_input_order() -> None:
    output_name, structure_id, _pdb_code, rank = _make_step02_output_name(
        Path("NEW2.model.pdb"),
        input_order=3,
        target_id="NEW2",
        step02_cfg={},
    )

    assert output_name == "NEW2_3_complex.pdb"
    assert structure_id == "NEW2_3"
    assert rank == "3"
