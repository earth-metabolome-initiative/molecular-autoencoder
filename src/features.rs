//! Molecular descriptor targets for auxiliary training tasks.

use elements_rs::Element;
use serde::{Deserialize, Serialize};
use smiles_parser::prelude::Smiles;

use crate::{Error, Result};

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

/// Inclusive-range quality filter applied during SMILES preprocessing.
///
/// Construct with [`SmilesQualityFilter::builder`]; bounds are validated at
/// `build()` time so inverted ranges fail fast instead of silently rejecting
/// every molecule. Each bound is optional; absent bounds impose no constraint.
/// A record is accepted only when every active bound is satisfied. Filters
/// operate on [`DescriptorTargets`] computed from the parsed SMILES, so
/// molecules rejected here never reach fingerprint computation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SmilesQualityFilter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_heavy_atoms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_heavy_atoms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_molecular_mass: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_molecular_mass: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_formal_charge: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_formal_charge: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_connected_components: Option<u32>,
}

impl SmilesQualityFilter {
    /// Starts a fluent builder for a quality filter.
    #[must_use]
    pub const fn builder() -> SmilesQualityFilterBuilder {
        SmilesQualityFilterBuilder::new()
    }

    /// Whether the filter has any active bound.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.min_heavy_atoms.is_some()
            || self.max_heavy_atoms.is_some()
            || self.min_molecular_mass.is_some()
            || self.max_molecular_mass.is_some()
            || self.min_formal_charge.is_some()
            || self.max_formal_charge.is_some()
            || self.max_connected_components.is_some()
    }

    /// Returns `true` when every active bound accepts these descriptors.
    #[must_use]
    pub fn accepts(&self, descriptors: &DescriptorTargets) -> bool {
        if let Some(min) = self.min_heavy_atoms
            && descriptors.heavy_atom_count < min
        {
            return false;
        }
        if let Some(max) = self.max_heavy_atoms
            && descriptors.heavy_atom_count > max
        {
            return false;
        }
        if let Some(min) = self.min_molecular_mass
            && descriptors.molecular_mass < min
        {
            return false;
        }
        if let Some(max) = self.max_molecular_mass
            && descriptors.molecular_mass > max
        {
            return false;
        }
        if let Some(min) = self.min_formal_charge
            && descriptors.formal_charge < min
        {
            return false;
        }
        if let Some(max) = self.max_formal_charge
            && descriptors.formal_charge > max
        {
            return false;
        }
        if let Some(max) = self.max_connected_components
            && descriptors.connected_component_count > max
        {
            return false;
        }
        true
    }

    /// Minimum heavy-atom bound, if set.
    #[must_use]
    pub const fn min_heavy_atoms(&self) -> Option<u32> {
        self.min_heavy_atoms
    }

    /// Maximum heavy-atom bound, if set.
    #[must_use]
    pub const fn max_heavy_atoms(&self) -> Option<u32> {
        self.max_heavy_atoms
    }

    /// Minimum molecular-mass bound, if set.
    #[must_use]
    pub const fn min_molecular_mass(&self) -> Option<f32> {
        self.min_molecular_mass
    }

    /// Maximum molecular-mass bound, if set.
    #[must_use]
    pub const fn max_molecular_mass(&self) -> Option<f32> {
        self.max_molecular_mass
    }

    /// Minimum formal-charge bound, if set.
    #[must_use]
    pub const fn min_formal_charge(&self) -> Option<i32> {
        self.min_formal_charge
    }

    /// Maximum formal-charge bound, if set.
    #[must_use]
    pub const fn max_formal_charge(&self) -> Option<i32> {
        self.max_formal_charge
    }

    /// Maximum connected-component bound, if set.
    #[must_use]
    pub const fn max_connected_components(&self) -> Option<u32> {
        self.max_connected_components
    }
}

/// Fluent builder for [`SmilesQualityFilter`].
///
/// All bounds are optional; unset bounds impose no constraint. Call
/// [`build`](Self::build) to validate ranges and produce an immutable filter.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SmilesQualityFilterBuilder {
    min_heavy_atoms: Option<u32>,
    max_heavy_atoms: Option<u32>,
    min_molecular_mass: Option<f32>,
    max_molecular_mass: Option<f32>,
    min_formal_charge: Option<i32>,
    max_formal_charge: Option<i32>,
    max_connected_components: Option<u32>,
}

impl SmilesQualityFilterBuilder {
    /// Creates a builder with no active bounds.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min_heavy_atoms: None,
            max_heavy_atoms: None,
            min_molecular_mass: None,
            max_molecular_mass: None,
            min_formal_charge: None,
            max_formal_charge: None,
            max_connected_components: None,
        }
    }

    /// Sets the minimum heavy-atom bound (non-hydrogen atoms, inclusive).
    #[must_use]
    pub const fn min_heavy_atoms(mut self, value: u32) -> Self {
        self.min_heavy_atoms = Some(value);
        self
    }

    /// Sets the maximum heavy-atom bound (non-hydrogen atoms, inclusive).
    #[must_use]
    pub const fn max_heavy_atoms(mut self, value: u32) -> Self {
        self.max_heavy_atoms = Some(value);
        self
    }

    /// Sets the minimum molecular-mass bound in atomic mass units.
    #[must_use]
    pub const fn min_molecular_mass(mut self, value: f32) -> Self {
        self.min_molecular_mass = Some(value);
        self
    }

    /// Sets the maximum molecular-mass bound in atomic mass units.
    #[must_use]
    pub const fn max_molecular_mass(mut self, value: f32) -> Self {
        self.max_molecular_mass = Some(value);
        self
    }

    /// Sets the minimum total formal charge.
    #[must_use]
    pub const fn min_formal_charge(mut self, value: i32) -> Self {
        self.min_formal_charge = Some(value);
        self
    }

    /// Sets the maximum total formal charge.
    #[must_use]
    pub const fn max_formal_charge(mut self, value: i32) -> Self {
        self.max_formal_charge = Some(value);
        self
    }

    /// Sets the maximum number of disconnected components; set to 1 to drop
    /// salts and mixtures.
    #[must_use]
    pub const fn max_connected_components(mut self, value: u32) -> Self {
        self.max_connected_components = Some(value);
        self
    }

    /// Validates the configured bounds and produces an immutable filter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when a `min_*` bound exceeds its
    /// corresponding `max_*` bound, when any mass bound is non-finite or
    /// negative, or when `max_connected_components` is zero.
    pub fn build(self) -> Result<SmilesQualityFilter> {
        if let (Some(min), Some(max)) = (self.min_heavy_atoms, self.max_heavy_atoms)
            && min > max
        {
            return Err(Error::ConfigInvalid {
                message: format!("min_heavy_atoms {min} exceeds max_heavy_atoms {max}"),
            });
        }
        if let Some(value) = self.min_molecular_mass
            && (!value.is_finite() || value < 0.0)
        {
            return Err(Error::ConfigInvalid {
                message: format!("min_molecular_mass must be finite and non-negative, got {value}"),
            });
        }
        if let Some(value) = self.max_molecular_mass
            && (!value.is_finite() || value < 0.0)
        {
            return Err(Error::ConfigInvalid {
                message: format!("max_molecular_mass must be finite and non-negative, got {value}"),
            });
        }
        if let (Some(min), Some(max)) = (self.min_molecular_mass, self.max_molecular_mass)
            && min > max
        {
            return Err(Error::ConfigInvalid {
                message: format!("min_molecular_mass {min} exceeds max_molecular_mass {max}"),
            });
        }
        if let (Some(min), Some(max)) = (self.min_formal_charge, self.max_formal_charge)
            && min > max
        {
            return Err(Error::ConfigInvalid {
                message: format!("min_formal_charge {min} exceeds max_formal_charge {max}"),
            });
        }
        if let Some(value) = self.max_connected_components
            && value == 0
        {
            return Err(Error::ConfigInvalid {
                message: "max_connected_components must be greater than zero".to_string(),
            });
        }

        Ok(SmilesQualityFilter {
            min_heavy_atoms: self.min_heavy_atoms,
            max_heavy_atoms: self.max_heavy_atoms,
            min_molecular_mass: self.min_molecular_mass,
            max_molecular_mass: self.max_molecular_mass,
            min_formal_charge: self.min_formal_charge,
            max_formal_charge: self.max_formal_charge,
            max_connected_components: self.max_connected_components,
        })
    }
}

/// Normalization used for scalar descriptor targets. Construct via
/// [`DescriptorConfig::builder`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_field_names)]
pub struct DescriptorConfig {
    mass_scale: f32,
    count_scale: f32,
    charge_scale: f32,
}

impl Default for DescriptorConfig {
    fn default() -> Self {
        DescriptorConfigBuilder::new()
            .build()
            .expect("default descriptor config is valid")
    }
}

impl DescriptorConfig {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> DescriptorConfigBuilder {
        DescriptorConfigBuilder::new()
    }

    /// Divisor applied to molecular mass.
    #[must_use]
    pub const fn mass_scale(&self) -> f32 {
        self.mass_scale
    }

    /// Divisor applied to non-charge counts.
    #[must_use]
    pub const fn count_scale(&self) -> f32 {
        self.count_scale
    }

    /// Divisor applied to formal charge.
    #[must_use]
    pub const fn charge_scale(&self) -> f32 {
        self.charge_scale
    }
}

/// Fluent builder for [`DescriptorConfig`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::struct_field_names)]
pub struct DescriptorConfigBuilder {
    mass_scale: f32,
    count_scale: f32,
    charge_scale: f32,
}

impl Default for DescriptorConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DescriptorConfigBuilder {
    /// Creates a builder seeded with the v1 normalization defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mass_scale: 1000.0,
            count_scale: 128.0,
            charge_scale: 8.0,
        }
    }

    /// Sets the molecular-mass divisor.
    #[must_use]
    pub const fn mass_scale(mut self, value: f32) -> Self {
        self.mass_scale = value;
        self
    }

    /// Sets the non-charge count divisor.
    #[must_use]
    pub const fn count_scale(mut self, value: f32) -> Self {
        self.count_scale = value;
        self
    }

    /// Sets the formal-charge divisor.
    #[must_use]
    pub const fn charge_scale(mut self, value: f32) -> Self {
        self.charge_scale = value;
        self
    }

    /// Validates the configured scales and builds the immutable config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when any scale is non-finite or
    /// non-positive (division by zero would produce non-finite targets).
    pub fn build(self) -> Result<DescriptorConfig> {
        for (label, value) in [
            ("mass_scale", self.mass_scale),
            ("count_scale", self.count_scale),
            ("charge_scale", self.charge_scale),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::ConfigInvalid {
                    message: format!("{label} must be finite and positive, got {value}"),
                });
            }
        }
        Ok(DescriptorConfig {
            mass_scale: self.mass_scale,
            count_scale: self.count_scale,
            charge_scale: self.charge_scale,
        })
    }
}

/// Molecule-derived descriptor targets. Constructed by
/// [`DescriptorTargets::from_smiles`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_field_names)]
pub struct DescriptorTargets {
    molecular_mass: f32,
    formal_charge: i32,
    carbon_count: u32,
    hydrogen_count: u32,
    nitrogen_count: u32,
    oxygen_count: u32,
    phosphorus_count: u32,
    sulfur_count: u32,
    fluorine_count: u32,
    chlorine_count: u32,
    bromine_count: u32,
    iodine_count: u32,
    heavy_atom_count: u32,
    total_hydrogen_count: u32,
    ring_atom_count: u32,
    ring_bond_count: u32,
    connected_component_count: u32,
    aromatic_atom_count: u32,
}

#[allow(missing_docs)]
impl DescriptorTargets {
    #[must_use]
    pub const fn molecular_mass(&self) -> f32 {
        self.molecular_mass
    }
    #[must_use]
    pub const fn formal_charge(&self) -> i32 {
        self.formal_charge
    }
    #[must_use]
    pub const fn carbon_count(&self) -> u32 {
        self.carbon_count
    }
    #[must_use]
    pub const fn hydrogen_count(&self) -> u32 {
        self.hydrogen_count
    }
    #[must_use]
    pub const fn nitrogen_count(&self) -> u32 {
        self.nitrogen_count
    }
    #[must_use]
    pub const fn oxygen_count(&self) -> u32 {
        self.oxygen_count
    }
    #[must_use]
    pub const fn phosphorus_count(&self) -> u32 {
        self.phosphorus_count
    }
    #[must_use]
    pub const fn sulfur_count(&self) -> u32 {
        self.sulfur_count
    }
    #[must_use]
    pub const fn fluorine_count(&self) -> u32 {
        self.fluorine_count
    }
    #[must_use]
    pub const fn chlorine_count(&self) -> u32 {
        self.chlorine_count
    }
    #[must_use]
    pub const fn bromine_count(&self) -> u32 {
        self.bromine_count
    }
    #[must_use]
    pub const fn iodine_count(&self) -> u32 {
        self.iodine_count
    }
    #[must_use]
    pub const fn heavy_atom_count(&self) -> u32 {
        self.heavy_atom_count
    }
    #[must_use]
    pub const fn total_hydrogen_count(&self) -> u32 {
        self.total_hydrogen_count
    }
    #[must_use]
    pub const fn ring_atom_count(&self) -> u32 {
        self.ring_atom_count
    }
    #[must_use]
    pub const fn ring_bond_count(&self) -> u32 {
        self.ring_bond_count
    }
    #[must_use]
    pub const fn connected_component_count(&self) -> u32 {
        self.connected_component_count
    }
    #[must_use]
    pub const fn aromatic_atom_count(&self) -> u32 {
        self.aromatic_atom_count
    }
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
        let count_scale = config.count_scale();
        [
            self.molecular_mass / config.mass_scale(),
            self.formal_charge as f32 / config.charge_scale(),
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
    fn quality_filter_default_is_inactive_and_accepts_everything() {
        let filter = SmilesQualityFilter::default();
        let smiles: Smiles = "CCO".parse().expect("valid SMILES");
        let descriptors = DescriptorTargets::from_smiles(&smiles);

        assert!(!filter.is_active());
        assert!(filter.accepts(&descriptors));
    }

    #[test]
    fn quality_filter_enforces_heavy_atom_bounds() {
        let descriptors = DescriptorTargets::from_smiles(&"CCO".parse().expect("valid")); // 3 heavy
        let too_small = SmilesQualityFilter::builder()
            .min_heavy_atoms(4)
            .build()
            .expect("valid filter");
        let too_large = SmilesQualityFilter::builder()
            .max_heavy_atoms(2)
            .build()
            .expect("valid filter");
        let exact = SmilesQualityFilter::builder()
            .min_heavy_atoms(3)
            .max_heavy_atoms(3)
            .build()
            .expect("valid filter");

        assert!(!too_small.accepts(&descriptors));
        assert!(!too_large.accepts(&descriptors));
        assert!(exact.accepts(&descriptors));
    }

    #[test]
    fn quality_filter_enforces_mass_bounds() {
        let descriptors = DescriptorTargets::from_smiles(&"CCO".parse().expect("valid"));
        let lower_floor = SmilesQualityFilter::builder()
            .min_molecular_mass(100.0)
            .build()
            .expect("valid filter");
        let upper_ceiling = SmilesQualityFilter::builder()
            .max_molecular_mass(10.0)
            .build()
            .expect("valid filter");
        let acceptable_window = SmilesQualityFilter::builder()
            .min_molecular_mass(10.0)
            .max_molecular_mass(500.0)
            .build()
            .expect("valid filter");

        assert!(!lower_floor.accepts(&descriptors));
        assert!(!upper_ceiling.accepts(&descriptors));
        assert!(acceptable_window.accepts(&descriptors));
    }

    #[test]
    fn quality_filter_enforces_formal_charge_bounds() {
        let neutral = DescriptorTargets::from_smiles(&"CCO".parse().expect("valid"));
        let cation =
            DescriptorTargets::from_smiles(&"[NH4+]".parse().expect("valid ammonium SMILES"));
        let neutral_only = SmilesQualityFilter::builder()
            .min_formal_charge(0)
            .max_formal_charge(0)
            .build()
            .expect("valid filter");

        assert!(neutral_only.accepts(&neutral));
        assert!(!neutral_only.accepts(&cation));
    }

    #[test]
    fn quality_filter_caps_connected_components() {
        let single = DescriptorTargets::from_smiles(&"CCO".parse().expect("valid"));
        let mixture = DescriptorTargets::from_smiles(&"CC.O".parse().expect("valid"));
        let one_component_only = SmilesQualityFilter::builder()
            .max_connected_components(1)
            .build()
            .expect("valid filter");

        assert!(one_component_only.accepts(&single));
        assert!(!one_component_only.accepts(&mixture));
    }

    #[test]
    fn quality_filter_builder_rejects_inverted_bounds() {
        let inverted_heavy = SmilesQualityFilter::builder()
            .min_heavy_atoms(50)
            .max_heavy_atoms(10)
            .build()
            .expect_err("inverted heavy-atom range must be rejected");
        assert!(matches!(
            inverted_heavy,
            crate::Error::ConfigInvalid { message } if message.contains("min_heavy_atoms")
        ));

        let inverted_mass = SmilesQualityFilter::builder()
            .min_molecular_mass(500.0)
            .max_molecular_mass(100.0)
            .build()
            .expect_err("inverted mass range must be rejected");
        assert!(matches!(
            inverted_mass,
            crate::Error::ConfigInvalid { message } if message.contains("molecular_mass")
        ));

        let zero_components = SmilesQualityFilter::builder()
            .max_connected_components(0)
            .build()
            .expect_err("max_connected_components=0 must be rejected");
        assert!(matches!(
            zero_components,
            crate::Error::ConfigInvalid { message } if message.contains("connected_components")
        ));
    }

    #[test]
    fn regression_target_order_and_scaling_are_stable() {
        let smiles: Smiles = "CCO".parse().expect("valid SMILES");
        let descriptors = DescriptorTargets::from_smiles(&smiles);
        let targets = descriptors.regression_targets(
            DescriptorConfig::builder()
                .mass_scale(1.0)
                .count_scale(1.0)
                .charge_scale(1.0)
                .build()
                .expect("valid descriptor config"),
        );

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
