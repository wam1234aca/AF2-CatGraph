#!/usr/bin/env python
# -*- coding: utf-8 -*-
'Rigid template-guided protein-ligand docking utilities. The workflow detects steric clashes, applies bounded deterministic rigid-body translations and rotations to the ligand, aligns target proteins to a template, and writes protein-ligand complexes.'

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import AlignMol


def load_pdb(path):
    'Load a PDB file as an RDKit molecule.'
    mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False, proximityBonding=False)
    if mol is None:
        raise ValueError(f"Could not load PDB file: {path}")
    return mol


def load_molecule(path):
    'Load a ligand molecule with format-specific fallbacks and tolerant sanitization.'

    file_ext = os.path.splitext(path)[1].lower()
    

    if file_ext == '.pdb':

        mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False, proximityBonding=False)
    elif file_ext in ['.mol', '.sdf']:

        mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
    else:

        mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
    
    if mol is None:
        return None
    

    try:
        Chem.SanitizeMol(mol, 
                        Chem.SanitizeFlags.SANITIZE_ALL ^ 
                        Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        print("Partial structure validation succeeded.")
    except Exception as e:
        print(f"Partial structure validation failed: {e}")
        print("Continuing with the incompletely validated structure.")
    
    return mol


def translate_molecule(mol, translation_vec):
    'Rigidly translate all atomic coordinates by the given vector.'
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, pos + translation_vec)


def rotate_molecule(mol, rotation_matrix, center=None):
    'Rigidly rotate a molecule about the specified center.'
    conf = mol.GetConformer()    

    if center is None:
        center = calculate_molecule_center(mol)   

    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))

        centered_pos = pos - center

        rotated_pos = np.dot(rotation_matrix, centered_pos)

        new_pos = rotated_pos + center
        conf.SetAtomPosition(i, new_pos)


def calculate_molecule_center(mol):
    'Return the geometric center of a molecule.'
    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        positions.append(np.array(conf.GetAtomPosition(i)))
    return np.mean(positions, axis=0)


def axis_angle_to_rotation_matrix(axis, angle):
    'Return a Rodrigues rotation matrix for an axis and angle.'

    axis = axis / np.linalg.norm(axis)    

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])    
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return rotation_matrix


def get_collision_severity(protein, ligand, threshold=2.0):
    'Return a weighted clash score, the dominant repulsion direction, and clashing ligand atoms.'
    conf_prot = protein.GetConformer()
    conf_lig = ligand.GetConformer()    
    collision_vectors = []
    collision_weights = []
    colliding_atoms = set()
    for i in range(ligand.GetNumAtoms()):
        pos_lig = np.array(conf_lig.GetAtomPosition(i))
        for j in range(protein.GetNumAtoms()):
            pos_prot = np.array(conf_prot.GetAtomPosition(j))
            dist = np.linalg.norm(pos_lig - pos_prot)            
            if dist < threshold:

                vec = pos_lig - pos_prot
                norm = np.linalg.norm(vec)
                if norm > 1e-6:

                    weight = (threshold - dist) / threshold
                    collision_vectors.append(vec / norm)
                    collision_weights.append(weight)
                    colliding_atoms.add(i)    

    collision_score = sum(collision_weights)    

    if collision_vectors:
        weights = np.array(collision_weights)
        vectors = np.array(collision_vectors)
        avg_direction = np.average(vectors, axis=0, weights=weights)

        norm = np.linalg.norm(avg_direction)
        if norm > 1e-6:
            avg_direction = avg_direction / norm
        else:
            avg_direction = np.zeros(3)
    else:
        avg_direction = np.zeros(3)
    
    return collision_score, avg_direction, colliding_atoms


def has_collision(mol1, mol2, threshold=2.0):
    'Return whether any interatomic distance is below the clash threshold.'
    score, _, _ = get_collision_severity(mol1, mol2, threshold)
    return score > 0


def avoid_collision_strategic(protein, ligand, max_attempts=50, 
                             collision_threshold=2.0,
                             max_translation=5.0,
                             max_rotation=30.0):
    'Reduce protein-ligand clashes with bounded deterministic rigid-body translations and rotations.'

    current_ligand = Chem.Mol(ligand)
    ligand_center = calculate_molecule_center(current_ligand)    

    total_translation = np.zeros(3)
    total_translation_distance = 0.0
    total_rotation_deg = 0.0   

    best_ligand = Chem.Mol(current_ligand)
    best_collision_score = float('inf')    

    last_collision_score = float('inf')    

    adjustment_phase = 0
    no_improvement_count = 0    
    for attempt in range(max_attempts):

        collision_score, collision_direction, colliding_atoms = get_collision_severity(
            protein, current_ligand, threshold=collision_threshold)        

        if collision_score == 0:
            print(f"Clash avoidance succeeded; total translation: {total_translation_distance:.2f} Å, "
                  f"total rotation: {total_rotation_deg:.1f}°, attempts: {attempt + 1}.")
            return current_ligand        

        if collision_score < best_collision_score:
            best_collision_score = collision_score
            best_ligand = Chem.Mol(current_ligand)       

        if collision_score >= last_collision_score:
            no_improvement_count += 1
        else:
            no_improvement_count = 0        

        if no_improvement_count >= 3:
            adjustment_phase = (adjustment_phase + 1) % 3
            no_improvement_count = 0            

        last_collision_score = collision_score        

        if adjustment_phase == 0:


            step_size = min(0.1 * (1 + attempt/10), 0.5)

            if total_translation_distance < max_translation:
                translation = collision_direction * step_size                

                new_total = np.linalg.norm(total_translation + translation)
                if new_total > max_translation:

                    scale = (max_translation - total_translation_distance) / step_size
                    if scale > 0:
                        translation = translation * scale
                    else:

                        adjustment_phase = 1
                        continue                

                translate_molecule(current_ligand, translation)
                total_translation += translation
                total_translation_distance = np.linalg.norm(total_translation)                
                print(f"Translation adjustment: {translation.tolist()}; total translation: {total_translation_distance:.2f} Å.")
            else:

                adjustment_phase = 1                
        elif adjustment_phase == 1:

            if total_rotation_deg < max_rotation:

                if np.all(collision_direction == 0):

                    rotation_axis = np.random.randn(3)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                else:


                    v1 = collision_direction
                    v2 = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(v1, v2)) > 0.9:
                        v2 = np.array([0.0, 1.0, 0.0])
                    rotation_axis = np.cross(v1, v2)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)                

                angle_deg = min(2.0 * (1 + attempt/20), 5.0)

                if total_rotation_deg + angle_deg > max_rotation:
                    angle_deg = max_rotation - total_rotation_deg
                    if angle_deg < 0.1:
                        adjustment_phase = 2
                        continue                

                angle_rad = np.radians(angle_deg)
                rotation_matrix = axis_angle_to_rotation_matrix(rotation_axis, angle_rad)
                rotate_molecule(current_ligand, rotation_matrix, center=ligand_center)
                total_rotation_deg += angle_deg                
                print(f"Rotation adjustment: axis={rotation_axis.tolist()}, angle={angle_deg:.2f}°, total rotation={total_rotation_deg:.2f}°.")
            else:

                adjustment_phase = 2                
        else:

            can_translate = total_translation_distance < max_translation
            can_rotate = total_rotation_deg < max_rotation            
            if not (can_translate or can_rotate):
                print("All adjustment limits were reached; optimization cannot continue.")
                break                
            if can_translate:

                step_size = min(0.05 * (1 + attempt/20), 0.2)
                translation = collision_direction * step_size                

                new_total = np.linalg.norm(total_translation + translation)
                if new_total <= max_translation:
                    translate_molecule(current_ligand, translation)
                    total_translation += translation
                    total_translation_distance = new_total            
            if can_rotate:

                if np.all(collision_direction == 0):
                    rotation_axis = np.random.randn(3)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                else:
                    v1 = collision_direction
                    v2 = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(v1, v2)) > 0.9:
                        v2 = np.array([0.0, 1.0, 0.0])
                    rotation_axis = np.cross(v1, v2)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)                

                angle_deg = min(1.0 * (1 + attempt/30), 3.0)                

                if total_rotation_deg + angle_deg <= max_rotation:
                    angle_rad = np.radians(angle_deg)
                    rotation_matrix = axis_angle_to_rotation_matrix(rotation_axis, angle_rad)

                    ligand_center = calculate_molecule_center(current_ligand)
                    rotate_molecule(current_ligand, rotation_matrix, center=ligand_center)
                    total_rotation_deg += angle_deg            
            print(f"Combined adjustment: translation={total_translation_distance:.2f} Å, rotation={total_rotation_deg:.2f}°.")    

    if best_collision_score < float('inf'):
        print(f"Maximum attempts reached; returning the best solution found (clash score: {best_collision_score:.2f}).")
        return best_ligand
    
    print("Clashes could not be avoided within the limits; continuing with the original ligand.")
    return ligand


def create_protein_atom_map(protein1, protein2):
    'Create a CA-atom index mapping between two protein structures.'

    ca_atoms1 = []
    ca_atoms2 = []
    

    for i in range(protein1.GetNumAtoms()):
        atom = protein1.GetAtomWithIdx(i)
        if atom.GetPDBResidueInfo() and atom.GetPDBResidueInfo().GetName().strip() == "CA":
            ca_atoms1.append(i)
    
    for i in range(protein2.GetNumAtoms()):
        atom = protein2.GetAtomWithIdx(i)
        if atom.GetPDBResidueInfo() and atom.GetPDBResidueInfo().GetName().strip() == "CA":
            ca_atoms2.append(i)
    

    min_atoms = min(len(ca_atoms1), len(ca_atoms2))
    if min_atoms == 0:
        print("No protein backbone CA atoms were found; attempting direct coordinate alignment.")
        return None
    

    atom_map = [(ca_atoms1[i], ca_atoms2[i]) for i in range(min_atoms)]
    print(f"Atom mapping created successfully for {min_atoms} CA atoms.")
    
    return atom_map


def align_proteins(protein1, protein2):
    'Align the target protein to the reference protein and return the RMSD.'

    atom_map = create_protein_atom_map(protein1, protein2)
    

    if atom_map:
        rmsd = AlignMol(protein1, protein2, atomMap=atom_map)
    else:

        try:
            rmsd = AlignMol(protein1, protein2)
            print("Using the default alignment method; the result may be imprecise.")
        except:

            print("Standard alignment failed; attempting centroid-based alignment.")
            center1 = calculate_molecule_center(protein1)
            center2 = calculate_molecule_center(protein2)

            translate_molecule(protein1, center2 - center1)
            rmsd = float('inf')
    
    return rmsd


def rigid_docking(template_protein, ligand, mutant_protein):
    'Align a target protein to the template, resolve ligand clashes, and return the combined complex.'

    rmsd = align_proteins(mutant_protein, template_protein)
    print(f"Protein-alignment RMSD: {rmsd:.3f} Å.")
    

    if has_collision(mutant_protein, ligand):
        print("Atomic clashes detected; starting deterministic clash avoidance...")

        ligand = avoid_collision_strategic(
            mutant_protein, 
            ligand, 
            max_attempts=50, 
            collision_threshold=2.0,
            max_translation=5.0,
            max_rotation=30.0
        )

    complex_mol = Chem.CombineMols(mutant_protein, ligand)
    return complex_mol


def main():

    ligand_file = "ligand.pdb"
    template_protein_file = "template.pdb"
    mutant_folder = "./mutants"
    output_folder = "./complex_output"
    # ==========================================
    os.makedirs(output_folder, exist_ok=True)
    print("Loading the template protein and ligand conformer...")
    template_protein = load_pdb(template_protein_file)
    

    ligand = load_molecule(ligand_file)
    
    if ligand is None:
        raise ValueError("Could not load the ligand file.")
        

    if ligand.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(ligand, randomSeed=42)
            print("Generated a 3D conformer for the ligand.")
        except Exception as e:
            print(f"Could not generate a ligand conformer: {e}")
            raise ValueError("Could not generate the required 3D ligand conformer.")
            

    pdb_files = [f for f in os.listdir(mutant_folder) if f.endswith(".pdb")]
    if not pdb_files:
        raise ValueError("No .pdb files were found in the ensemble directory.")    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(mutant_folder, pdb_file)
        print(f"\nProcessing ensemble structure: {pdb_file}")
        mutant_protein = load_pdb(pdb_path)
        complex_mol = rigid_docking(template_protein, ligand, mutant_protein)        
        out_name = os.path.splitext(pdb_file)[0] + "_complex.pdb"
        out_path = os.path.join(output_folder, out_name)

        with Chem.PDBWriter(out_path) as writer:
            writer.write(complex_mol)
        print(f"Complex saved: {out_path}")


if __name__ == "__main__":
    main()
