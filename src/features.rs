//! Molecular descriptor targets for auxiliary training tasks.

use elements_rs::Element;
use serde::{Deserialize, Serialize};
use smiles_parser::prelude::Smiles;

/// Regression target labels in the order emitted by [`DescriptorTargets`].
pub const REGRESSION_TARGET_LABELS: [&str; REGRESSION_TARGET_WIDTH] = [
    "molecular_mass",
    "formal_charge",
    "count_c",
    "count_h",
    "count_n",
    "count_o",
    "count_p",
    "count_s",
    "count_f",
    "count_cl",
    "count_br",
    "count_i",
    "heavy_atom_count",
    "total_hydrogen_count",
    "ring_atom_count",
    "ring_bond_count",
    "component_count",
    "aromatic_atom_count",
];

/// Number of scalar regression descriptors.
pub const REGRESSION_TARGET_WIDTH: usize = 18;

/// Normalization used for scalar descriptor targets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DescriptorConfig {
    /// Divisor applied to molecular mass.
    pub mass_scale: f32,
    /// Divisor applied to non-charge counts.
    pub count_scale: f32,
    /// Divisor applied to formal charge.
    pub charge_scale: f32,
}

impl Default for DescriptorConfig {
    fn default() -> Self {
        Self {
            mass_scale: 1000.0,
            count_scale: 128.0,
            charge_scale: 8.0,
        }
    }
}

/// Molecule-derived descriptor targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DescriptorTargets {
    /// Sum of standard atomic weights including explicit and implicit hydrogens.
    pub molecular_mass: f32,
    /// Total formal charge.
    pub formal_charge: i32,
    /// Carbon count.
    pub carbon_count: u32,
    /// Hydrogen count, including explicit and implicit hydrogens.
    pub hydrogen_count: u32,
    /// Nitrogen count.
    pub nitrogen_count: u32,
    /// Oxygen count.
    pub oxygen_count: u32,
    /// Phosphorus count.
    pub phosphorus_count: u32,
    /// Sulfur count.
    pub sulfur_count: u32,
    /// Fluorine count.
    pub fluorine_count: u32,
    /// Chlorine count.
    pub chlorine_count: u32,
    /// Bromine count.
    pub bromine_count: u32,
    /// Iodine count.
    pub iodine_count: u32,
    /// Non-hydrogen atom count.
    pub heavy_atom_count: u32,
    /// Explicit plus implicit hydrogen count.
    pub total_hydrogen_count: u32,
    /// Number of atoms in at least one ring.
    pub ring_atom_count: u32,
    /// Number of bonds in at least one ring.
    pub ring_bond_count: u32,
    /// Number of disconnected components.
    pub connected_component_count: u32,
    /// Number of atoms aromatic under the default aromaticity model.
    pub aromatic_atom_count: u32,
}

impl DescriptorTargets {
    /// Computes descriptor targets from a parsed SMILES graph.
    #[must_use]
    pub fn from_smiles(smiles: &Smiles) -> Self {
        let mut molecular_mass = 0.0_f32;
        let mut formal_charge = 0_i32;
        let mut carbon_count = 0_u32;
        let mut hydrogen_count = 0_u32;
        let mut nitrogen_count = 0_u32;
        let mut oxygen_count = 0_u32;
        let mut phosphorus_count = 0_u32;
        let mut sulfur_count = 0_u32;
        let mut fluorine_count = 0_u32;
        let mut chlorine_count = 0_u32;
        let mut bromine_count = 0_u32;
        let mut iodine_count = 0_u32;
        let mut heavy_atom_count = 0_u32;
        let mut total_hydrogen_count = 0_u32;

        for (atom_id, atom) in smiles.nodes().iter().enumerate() {
            formal_charge += i32::from(atom.charge_value());
            let attached_hydrogens = u32::from(atom.hydrogen_count())
                + u32::from(smiles.implicit_hydrogen_count(atom_id));
            total_hydrogen_count += attached_hydrogens;
            hydrogen_count += attached_hydrogens;
            molecular_mass +=
                attached_hydrogens as f32 * Element::H.standard_atomic_weight() as f32;

            if let Some(element) = atom.element() {
                molecular_mass += element.standard_atomic_weight() as f32;
                match element {
                    Element::C => carbon_count += 1,
                    Element::H => {
                        hydrogen_count += 1;
                        total_hydrogen_count += 1;
                    }
                    Element::N => nitrogen_count += 1,
                    Element::O => oxygen_count += 1,
                    Element::P => phosphorus_count += 1,
                    Element::S => sulfur_count += 1,
                    Element::F => fluorine_count += 1,
                    Element::Cl => chlorine_count += 1,
                    Element::Br => bromine_count += 1,
                    Element::I => iodine_count += 1,
                    _ => {}
                }
                if element != Element::H {
                    heavy_atom_count += 1;
                }
            }
        }

        let ring_membership = smiles.ring_membership();
        let connected_component_count = smiles.connected_components().number_of_components() as u32;
        let aromatic_atom_count = smiles.aromaticity_assignment().atom_ids().len() as u32;

        Self {
            molecular_mass,
            formal_charge,
            carbon_count,
            hydrogen_count,
            nitrogen_count,
            oxygen_count,
            phosphorus_count,
            sulfur_count,
            fluorine_count,
            chlorine_count,
            bromine_count,
            iodine_count,
            heavy_atom_count,
            total_hydrogen_count,
            ring_atom_count: ring_membership.atom_ids().len() as u32,
            ring_bond_count: ring_membership.bond_edges().len() as u32,
            connected_component_count,
            aromatic_atom_count,
        }
    }

    /// Returns normalized scalar regression targets.
    #[must_use]
    pub fn regression_targets(&self, config: DescriptorConfig) -> [f32; REGRESSION_TARGET_WIDTH] {
        let count_scale = config.count_scale;
        [
            self.molecular_mass / config.mass_scale,
            self.formal_charge as f32 / config.charge_scale,
            self.carbon_count as f32 / count_scale,
            self.hydrogen_count as f32 / count_scale,
            self.nitrogen_count as f32 / count_scale,
            self.oxygen_count as f32 / count_scale,
            self.phosphorus_count as f32 / count_scale,
            self.sulfur_count as f32 / count_scale,
            self.fluorine_count as f32 / count_scale,
            self.chlorine_count as f32 / count_scale,
            self.bromine_count as f32 / count_scale,
            self.iodine_count as f32 / count_scale,
            self.heavy_atom_count as f32 / count_scale,
            self.total_hydrogen_count as f32 / count_scale,
            self.ring_atom_count as f32 / count_scale,
            self.ring_bond_count as f32 / count_scale,
            self.connected_component_count as f32 / count_scale,
            self.aromatic_atom_count as f32 / count_scale,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ethanol_descriptors_include_implicit_hydrogens() {
        let smiles: Smiles = "CCO".parse().expect("valid SMILES");
        let descriptors = DescriptorTargets::from_smiles(&smiles);

        assert_eq!(descriptors.carbon_count, 2);
        assert_eq!(descriptors.oxygen_count, 1);
        assert_eq!(descriptors.hydrogen_count, 6);
        assert_eq!(descriptors.total_hydrogen_count, 6);
        assert_eq!(descriptors.heavy_atom_count, 3);
        assert_eq!(descriptors.connected_component_count, 1);
        assert!((descriptors.molecular_mass - 46.069).abs() < 0.01);
    }

    #[test]
    fn benzene_has_ring_and_aromatic_targets() {
        let smiles: Smiles = "c1ccccc1".parse().expect("valid SMILES");
        let descriptors = DescriptorTargets::from_smiles(&smiles);

        assert_eq!(descriptors.ring_atom_count, 6);
        assert_eq!(descriptors.ring_bond_count, 6);
        assert_eq!(descriptors.aromatic_atom_count, 6);
    }

    #[test]
    fn descriptors_capture_charge_and_disconnected_fragments() {
        let ammonium: Smiles = "[NH4+]".parse().expect("valid SMILES");
        let ammonium = DescriptorTargets::from_smiles(&ammonium);

        assert_eq!(ammonium.formal_charge, 1);
        assert_eq!(ammonium.nitrogen_count, 1);
        assert_eq!(ammonium.hydrogen_count, 4);
        assert_eq!(ammonium.connected_component_count, 1);

        let fragments: Smiles = "CC.O".parse().expect("valid SMILES");
        let fragments = DescriptorTargets::from_smiles(&fragments);
        assert_eq!(fragments.connected_component_count, 2);
        assert_eq!(fragments.carbon_count, 2);
        assert_eq!(fragments.oxygen_count, 1);
    }

    #[test]
    fn descriptors_capture_common_heteroatom_counts() {
        let smiles: Smiles = "[F].[Cl].[Br].[I].[P].[S].[N].[O]"
            .parse()
            .expect("valid SMILES");
        let descriptors = DescriptorTargets::from_smiles(&smiles);

        assert_eq!(descriptors.fluorine_count, 1);
        assert_eq!(descriptors.chlorine_count, 1);
        assert_eq!(descriptors.bromine_count, 1);
        assert_eq!(descriptors.iodine_count, 1);
        assert_eq!(descriptors.phosphorus_count, 1);
        assert_eq!(descriptors.sulfur_count, 1);
        assert_eq!(descriptors.nitrogen_count, 1);
        assert_eq!(descriptors.oxygen_count, 1);
        assert_eq!(descriptors.heavy_atom_count, 8);
        assert_eq!(descriptors.connected_component_count, 8);
    }

    #[test]
    fn regression_target_order_and_scaling_are_stable() {
        let smiles: Smiles = "CCO".parse().expect("valid SMILES");
        let descriptors = DescriptorTargets::from_smiles(&smiles);
        let targets = descriptors.regression_targets(DescriptorConfig {
            mass_scale: 1.0,
            count_scale: 1.0,
            charge_scale: 1.0,
        });

        assert_eq!(REGRESSION_TARGET_LABELS.len(), REGRESSION_TARGET_WIDTH);
        assert!((targets[0] - descriptors.molecular_mass).abs() < 1.0e-6);
        assert_eq!(targets[1], 0.0);
        assert_eq!(targets[2], 2.0);
        assert_eq!(targets[3], 6.0);
        assert_eq!(targets[5], 1.0);
        assert_eq!(targets[12], 3.0);
        assert_eq!(targets[13], 6.0);
        assert_eq!(targets[16], 1.0);
    }
}
