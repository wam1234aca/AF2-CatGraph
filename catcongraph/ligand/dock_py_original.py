#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化后的蛋白-配体刚性对接流程：
1. 加载蛋白PDB文件和小分子结构；
2. 对小分子执行碰撞检测，如果与蛋白发生碰撞（原子间距离小于给定阈值），
   则采用基于确定性策略的平移和旋转组合算法，在限制区域内调整小分子位置；
3. 利用刚性对齐将突变体蛋白对齐至模板蛋白，并将调整后的小分子插入到复合物中；
4. 最后保存复合体的PDB文件。

本算法特点：
- 保持完全刚性对接（配体整体结构不变）
- 限制配体移动和旋转范围（模拟盒子限制）
- 采用确定性策略而非随机策略
- 优先采用最小程度的调整解决碰撞
"""

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import AlignMol


def load_pdb(path):
    """
    加载PDB文件为RDKit分子对象。
    参数:
        path: PDB文件路径
    返回:
        RDKit的分子对象
    """
    mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False, proximityBonding=False)
    if mol is None:
        raise ValueError(f"无法加载PDB文件: {path}")
    return mol


def load_molecule(path):
    """
    加载小分子文件为RDKit分子对象，包含错误处理和多种加载策略。
    参数:
        path: 小分子文件路径
    返回:
        RDKit的分子对象
    """
    # 获取文件扩展名
    file_ext = os.path.splitext(path)[1].lower()
    
    # 根据文件扩展名选择合适的加载函数
    if file_ext == '.pdb':
        # PDB文件加载
        mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False, proximityBonding=False)
    elif file_ext in ['.mol', '.sdf']:
        # MOL/SDF文件加载
        mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
    else:
        # 其他格式文件，尝试通过MOL格式加载
        mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
    
    if mol is None:
        return None
    
    # 尝试部分结构验证，忽略价态检查
    try:
        Chem.SanitizeMol(mol, 
                        Chem.SanitizeFlags.SANITIZE_ALL ^ 
                        Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        print("✅ 成功进行部分结构验证")
    except Exception as e:
        print(f"⚠️ 部分结构验证失败: {e}")
        print("⚠️ 继续使用未完全验证的结构")
    
    return mol


def translate_molecule(mol, translation_vec):
    """
    将分子整体刚性平移一个向量。
    参数:
        mol: RDKit分子对象
        translation_vec: 形如numpy数组的平移向量
    """
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, pos + translation_vec)


def rotate_molecule(mol, rotation_matrix, center=None):
    """
    将分子绕指定中心点进行刚性旋转。    
    参数:
        mol: RDKit分子对象
        rotation_matrix: 3x3旋转矩阵
        center: 旋转中心点坐标，如果为None则使用分子的几何中心
    """
    conf = mol.GetConformer()    
    # 如果未指定旋转中心，则使用分子的几何中心
    if center is None:
        center = calculate_molecule_center(mol)   
    # 对每个原子应用旋转
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        # 将原子坐标移到原点（相对于旋转中心）
        centered_pos = pos - center
        # 应用旋转
        rotated_pos = np.dot(rotation_matrix, centered_pos)
        # 移回原位置
        new_pos = rotated_pos + center
        conf.SetAtomPosition(i, new_pos)


def calculate_molecule_center(mol):
    """
    计算分子的几何中心。    
    参数:
        mol: RDKit的分子对象
    
    返回:
        分子的几何中心坐标（3D向量）
    """
    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        positions.append(np.array(conf.GetAtomPosition(i)))
    return np.mean(positions, axis=0)


def axis_angle_to_rotation_matrix(axis, angle):
    """
    根据旋转轴和角度生成旋转矩阵（罗德里格斯公式）。    
    参数:
        axis: 旋转轴（单位向量）
        angle: 旋转角度（弧度）
    
    返回:
        3x3旋转矩阵
    """
    # 确保轴是单位向量
    axis = axis / np.linalg.norm(axis)    
    # 罗德里格斯公式
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])    
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return rotation_matrix


def get_collision_severity(protein, ligand, threshold=2.0):
    """
    计算碰撞的严重程度和最主要的碰撞方向。    
    参数:
        protein, ligand: RDKit分子对象
        threshold: 碰撞距离阈值    
    返回:
        collision_score: 碰撞严重度分数（碰撞对数量的加权和）
        avg_direction: 主要碰撞方向（单位向量）
        colliding_atoms: 发生碰撞的配体原子索引集合
    """
    conf_prot = protein.GetConformer()
    conf_lig = ligand.GetConformer()    
    collision_vectors = []  # 碰撞斥力向量
    collision_weights = []  # 碰撞权重（越近权重越大）
    colliding_atoms = set()  # 发生碰撞的配体原子    
    for i in range(ligand.GetNumAtoms()):
        pos_lig = np.array(conf_lig.GetAtomPosition(i))
        for j in range(protein.GetNumAtoms()):
            pos_prot = np.array(conf_prot.GetAtomPosition(j))
            dist = np.linalg.norm(pos_lig - pos_prot)            
            if dist < threshold:
                # 计算碰撞向量（配体原子到蛋白原子的方向的反向）
                vec = pos_lig - pos_prot
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    # 距离越小，权重越大
                    weight = (threshold - dist) / threshold
                    collision_vectors.append(vec / norm)
                    collision_weights.append(weight)
                    colliding_atoms.add(i)    
    # 计算总碰撞得分（所有碰撞权重的总和）
    collision_score = sum(collision_weights)    
    # 计算加权平均碰撞方向
    if collision_vectors:
        weights = np.array(collision_weights)
        vectors = np.array(collision_vectors)
        avg_direction = np.average(vectors, axis=0, weights=weights)
        # 归一化
        norm = np.linalg.norm(avg_direction)
        if norm > 1e-6:
            avg_direction = avg_direction / norm
        else:
            avg_direction = np.zeros(3)
    else:
        avg_direction = np.zeros(3)
    
    return collision_score, avg_direction, colliding_atoms


def has_collision(mol1, mol2, threshold=2.0):
    """
    检测两个分子是否存在原子碰撞（原子间距离小于阈值）。
    参数:
        mol1, mol2: RDKit分子对象
        threshold: 判定碰撞的距离阈值（单位Å）
    返回:
        True: 存在碰撞； False: 无碰撞。
    """
    score, _, _ = get_collision_severity(mol1, mol2, threshold)
    return score > 0


def avoid_collision_strategic(protein, ligand, max_attempts=50, 
                             collision_threshold=2.0,
                             max_translation=5.0,  # 最大总平移距离（Å）
                             max_rotation=30.0):   # 最大总旋转角度（度）
    """
    使用确定性策略调整小分子位置和旋转角度，避免与蛋白碰撞。
    遵循"盒子限制"概念，限制配体移动和旋转的最大范围。    
    策略：
    1. 尝试轻微平移
    2. 如果效果不佳，尝试轻微旋转
    3. 如果仍不足够，同时应用平移和旋转
    4. 随着迭代进行，逐渐增加调整幅度但保持在限制范围内    
    参数:
        protein: RDKit的蛋白分子对象
        ligand: RDKit的小分子分子对象
        max_attempts: 最大尝试次数
        collision_threshold: 碰撞距离阈值（Å）
        max_translation: 最大允许的总平移距离（Å）
        max_rotation: 最大允许的总旋转角度（度）
        
    返回:
        调整后的小分子对象
    """
    # 创建小分子的拷贝用于调整
    current_ligand = Chem.Mol(ligand)
    ligand_center = calculate_molecule_center(current_ligand)    
    # 跟踪总平移和总旋转
    total_translation = np.zeros(3)
    total_translation_distance = 0.0
    total_rotation_deg = 0.0   
    # 记录最佳状态（碰撞最小的状态）
    best_ligand = Chem.Mol(current_ligand)
    best_collision_score = float('inf')    
    # 上一次的碰撞得分，用于评估是否有改善
    last_collision_score = float('inf')    
    # 调整阶段标志: 0=仅平移, 1=仅旋转, 2=平移+旋转
    adjustment_phase = 0
    no_improvement_count = 0    
    for attempt in range(max_attempts):
        # 检查当前碰撞情况
        collision_score, collision_direction, colliding_atoms = get_collision_severity(
            protein, current_ligand, threshold=collision_threshold)        
        # 如果没有碰撞，则成功完成
        if collision_score == 0:
            print(f"✅ 避免碰撞成功，总平移距离: {total_translation_distance:.2f}Å，"
                  f"总旋转角度: {total_rotation_deg:.1f}°，尝试次数: {attempt + 1}")
            return current_ligand        
        # 如果当前状态比之前记录的最佳状态更好，则更新最佳状态
        if collision_score < best_collision_score:
            best_collision_score = collision_score
            best_ligand = Chem.Mol(current_ligand)       
        # 检查是否有改善
        if collision_score >= last_collision_score:
            no_improvement_count += 1
        else:
            no_improvement_count = 0        
        # 如果连续多次没有改善，切换调整策略
        if no_improvement_count >= 3:
            adjustment_phase = (adjustment_phase + 1) % 3
            no_improvement_count = 0            
        # 记录当前碰撞得分
        last_collision_score = collision_score        
        # 根据当前阶段选择调整方式
        if adjustment_phase == 0:  # 仅平移
            # 计算平移向量 - 沿碰撞方向相反方向平移
            # 平移距离根据迭代进程动态调整，但保持较小
            step_size = min(0.1 * (1 + attempt/10), 0.5)  # 从0.1逐渐增加但不超过0.5Å            
            # 如果还有平移空间
            if total_translation_distance < max_translation:
                translation = collision_direction * step_size                
                # 检查是否会超出总平移限制
                new_total = np.linalg.norm(total_translation + translation)
                if new_total > max_translation:
                    # 调整平移量使总平移不超过限制
                    scale = (max_translation - total_translation_distance) / step_size
                    if scale > 0:
                        translation = translation * scale
                    else:
                        # 无法再平移，切换到旋转阶段
                        adjustment_phase = 1
                        continue                
                # 应用平移
                translate_molecule(current_ligand, translation)
                total_translation += translation
                total_translation_distance = np.linalg.norm(total_translation)                
                print(f"平移调整: {translation.tolist()}, 总平移距离: {total_translation_distance:.2f}Å")
            else:
                # 达到平移限制，切换到旋转
                adjustment_phase = 1                
        elif adjustment_phase == 1:  # 仅旋转
            # 如果还有旋转空间
            if total_rotation_deg < max_rotation:
                # 确定旋转轴 - 垂直于主碰撞方向
                if np.all(collision_direction == 0):
                    # 如果没有明确的碰撞方向，使用随机轴
                    rotation_axis = np.random.randn(3)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                else:
                    # 找到垂直于碰撞方向的轴
                    # 使用正交向量计算
                    v1 = collision_direction
                    v2 = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(v1, v2)) > 0.9:
                        v2 = np.array([0.0, 1.0, 0.0])
                    rotation_axis = np.cross(v1, v2)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)                
                # 旋转角度根据迭代进程动态调整，但保持较小
                angle_deg = min(2.0 * (1 + attempt/20), 5.0)  # 从2度逐渐增加但不超过5度                
                # 检查是否会超出总旋转限制
                if total_rotation_deg + angle_deg > max_rotation:
                    angle_deg = max_rotation - total_rotation_deg
                    if angle_deg < 0.1:  # 如果剩余角度太小，切换到组合阶段
                        adjustment_phase = 2
                        continue                
                # 应用旋转
                angle_rad = np.radians(angle_deg)
                rotation_matrix = axis_angle_to_rotation_matrix(rotation_axis, angle_rad)
                rotate_molecule(current_ligand, rotation_matrix, center=ligand_center)
                total_rotation_deg += angle_deg                
                print(f"旋转调整: 轴={rotation_axis.tolist()}, 角度={angle_deg:.2f}°, 总旋转={total_rotation_deg:.2f}°")
            else:
                # 达到旋转限制，切换到组合调整
                adjustment_phase = 2                
        else:  # 平移+旋转组合
            # 结合平移和旋转，但都用较小幅度
            can_translate = total_translation_distance < max_translation
            can_rotate = total_rotation_deg < max_rotation            
            if not (can_translate or can_rotate):
                print("⚠️ 已达到所有调整限制，无法继续优化")
                break                
            if can_translate:
                # 小幅平移
                step_size = min(0.05 * (1 + attempt/20), 0.2)
                translation = collision_direction * step_size                
                # 检查平移限制
                new_total = np.linalg.norm(total_translation + translation)
                if new_total <= max_translation:
                    translate_molecule(current_ligand, translation)
                    total_translation += translation
                    total_translation_distance = new_total            
            if can_rotate:
                # 决定旋转轴 - 使用交叉乘积找到垂直于碰撞方向的轴
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
                # 小幅旋转
                angle_deg = min(1.0 * (1 + attempt/30), 3.0)                
                # 检查旋转限制
                if total_rotation_deg + angle_deg <= max_rotation:
                    angle_rad = np.radians(angle_deg)
                    rotation_matrix = axis_angle_to_rotation_matrix(rotation_axis, angle_rad)
                    # 可能需要更新配体中心（如果平移过）
                    ligand_center = calculate_molecule_center(current_ligand)
                    rotate_molecule(current_ligand, rotation_matrix, center=ligand_center)
                    total_rotation_deg += angle_deg            
            print(f"组合调整: 平移距离={total_translation_distance:.2f}Å, 旋转角度={total_rotation_deg:.2f}°")    
    # 如果达到最大尝试次数，返回最佳找到的解决方案
    if best_collision_score < float('inf'):
        print(f"⚠️ 达到最大尝试次数，返回找到的最佳解决方案（碰撞得分: {best_collision_score:.2f}）")
        return best_ligand
    
    print("⚠️ 未能在限制范围内避免碰撞，继续使用原始配体")
    return ligand


def create_protein_atom_map(protein1, protein2):
    """
    创建两个蛋白质结构之间的原子映射。
    策略：基于蛋白质骨架原子(CA,N,C,O)的对应位置创建映射。
    
    参数:
        protein1, protein2: RDKit分子对象
        
    返回:
        原子映射列表 [(idx1, idx2), ...]，如果无法创建映射则返回None
    """
    # 获取蛋白质中所有的CA原子(Alpha碳)
    ca_atoms1 = []
    ca_atoms2 = []
    
    # 查找骨架原子CA
    for i in range(protein1.GetNumAtoms()):
        atom = protein1.GetAtomWithIdx(i)
        if atom.GetPDBResidueInfo() and atom.GetPDBResidueInfo().GetName().strip() == "CA":
            ca_atoms1.append(i)
    
    for i in range(protein2.GetNumAtoms()):
        atom = protein2.GetAtomWithIdx(i)
        if atom.GetPDBResidueInfo() and atom.GetPDBResidueInfo().GetName().strip() == "CA":
            ca_atoms2.append(i)
    
    # 如果CA原子数量不同，尝试使用最小值
    min_atoms = min(len(ca_atoms1), len(ca_atoms2))
    if min_atoms == 0:
        print("⚠️ 无法找到蛋白质骨架CA原子，尝试使用坐标直接对齐")
        return None
    
    # 创建原子映射
    atom_map = [(ca_atoms1[i], ca_atoms2[i]) for i in range(min_atoms)]
    print(f"✅ 成功创建原子映射，共映射了{min_atoms}个CA原子")
    
    return atom_map


def align_proteins(protein1, protein2):
    """
    对齐两个蛋白质结构。
    
    参数:
        protein1: 目标蛋白 (将被移动到与protein2对齐)
        protein2: 参考蛋白
        
    返回:
        对齐的RMSD值
    """
    # 创建原子映射
    atom_map = create_protein_atom_map(protein1, protein2)
    
    # 如果有可用的原子映射，使用它进行对齐
    if atom_map:
        rmsd = AlignMol(protein1, protein2, atomMap=atom_map)
    else:
        # 如果没有原子映射，尝试使用所有重原子对齐（不可靠）
        try:
            rmsd = AlignMol(protein1, protein2)
            print("⚠️ 使用默认方法对齐，结果可能不准确")
        except:
            # 作为最后手段，尝试基于质心和主轴对齐
            print("⚠️ 标准对齐失败，尝试基于质心对齐")
            center1 = calculate_molecule_center(protein1)
            center2 = calculate_molecule_center(protein2)
            # 平移到相同质心
            translate_molecule(protein1, center2 - center1)
            rmsd = float('inf')  # 无法计算真正的RMSD
    
    return rmsd


def rigid_docking(template_protein, ligand, mutant_protein):
    """
    通过对齐突变体蛋白至模板蛋白，并将小分子插入其中构建复合物。
    调用avoid_collision_strategic对小分子进行确定性的碰撞优化。    
    参数:
        template_protein: 模板蛋白（RDKit分子对象）
        ligand: 小分子配体（RDKit分子对象）
        mutant_protein: 突变体蛋白（RDKit分子对象）        
    返回:
        组合后的复合物（RDKit分子对象）
    """
    # 使用增强的对齐函数将突变体蛋白对齐至模板蛋白
    rmsd = align_proteins(mutant_protein, template_protein)
    print(f"📏 蛋白质对齐RMSD: {rmsd:.3f}Å")
    
    # 检查是否与小分子发生碰撞，若有，则尝试自动避碰
    if has_collision(mutant_protein, ligand):
        print("🛑 检测到原子碰撞，开始尝试确定性策略避碰...")
        # 使用确定性策略进行碰撞避免
        ligand = avoid_collision_strategic(
            mutant_protein, 
            ligand, 
            max_attempts=50, 
            collision_threshold=2.0,
            max_translation=5.0,  # 最大平移限制（Å）
            max_rotation=30.0     # 最大旋转限制（度）
        )
    # 合并蛋白和小分子构成复合物
    complex_mol = Chem.CombineMols(mutant_protein, ligand)
    return complex_mol


def main():
    # ========== 请设置你的文件路径 ==========
    ligand_file = "ligand.pdb"               # 小分子文件路径
    template_protein_file = "template.pdb"   # 模板蛋白PDB文件
    mutant_folder = "./mutants"              # 存放突变体蛋白PDB文件的文件夹
    output_folder = "./complex_output"       # 复合物文件保存路径
    # ==========================================
    os.makedirs(output_folder, exist_ok=True)
    print("🚀 加载模板蛋白和小分子构象...")
    template_protein = load_pdb(template_protein_file)
    
    # 使用新的load_molecule函数加载小分子，处理价态问题
    ligand = load_molecule(ligand_file)
    
    if ligand is None:
        raise ValueError("无法加载小分子文件")
        
    # 若小分子缺少构象信息，则生成构象
    if ligand.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(ligand, randomSeed=42)
            print("✅ 为小分子生成了3D构象")
        except Exception as e:
            print(f"⚠️ 无法为小分子生成构象: {e}")
            raise ValueError("无法为小分子生成所需的3D构象")
            
    # 列出突变体PDB文件
    pdb_files = [f for f in os.listdir(mutant_folder) if f.endswith(".pdb")]
    if not pdb_files:
        raise ValueError("未在突变体文件夹中找到任何 .pdb 文件")    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(mutant_folder, pdb_file)
        print(f"\n📦 处理突变体结构: {pdb_file}")
        mutant_protein = load_pdb(pdb_path)
        complex_mol = rigid_docking(template_protein, ligand, mutant_protein)        
        out_name = os.path.splitext(pdb_file)[0] + "_complex.pdb"
        out_path = os.path.join(output_folder, out_name)
        # 写出PDB文件
        with Chem.PDBWriter(out_path) as writer:
            writer.write(complex_mol)
        print(f"✅ 复合物已保存: {out_path}")


if __name__ == "__main__":
    main()
