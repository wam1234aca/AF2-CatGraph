from catcongraph.gui_results import viewer_html


PDB_WITH_LIGAND = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C
HETATM    2  C1  LIG L   1       0.000   3.000   0.000  1.00 20.00           C
END
"""


def test_offline_viewer_matches_the_working_legacy_initialization() -> None:
    page, status = viewer_html(PDB_WITH_LIGAND, [], "Class 1")

    assert "const viewer = $3Dmol.createViewer('viewer'" in page
    assert "cartoon: {color: 'spectrum', opacity: 0.88}" in page
    assert "stick: {radius: 0.20, colorscheme: 'Jmol'}" in page
    assert "sphere: {scale: 0.22, colorscheme: 'Jmol'}" in page
    assert "viewer.addCylinder" in page
    assert "viewer.addLabel" in page
    assert "Canvas compatibility mode" not in page
    assert "Showing 1 ligand entity type" in status
