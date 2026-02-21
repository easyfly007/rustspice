/// # How to Add a New Simulator Option
///
/// 1. Add an entry to the `OPTION_DEFS` array below:
///    ```ignore
///    OptionDef {
///        name: "myoption",
///        description: "What this option controls",
///        default: OptionValue::Float(1.0),
///        range: OptionRange::FloatRange(0.0, 100.0),
///    },
///    ```
///
/// 2. Access it anywhere you have a `SimOptions` reference:
///    ```ignore
///    let val = options.get_float("myoption");
///    if options.is_set("myoption") { /* user explicitly set it */ }
///    ```
///
/// That's it. The parsing, validation, range checking, duplicate
/// warnings, and startup printing are all handled automatically.
///
/// # Performance Note
///
/// Options are stored in a `HashMap<String, OptionEntry>`. With a small
/// number of options (tens), this is perfectly fine — lookups are O(1)
/// and the hash overhead on short key strings is negligible. Options are
/// read once during setup (e.g. building `NewtonConfig`), not inside
/// any hot simulation loop.
///
/// If the option count grows large (hundreds+), consider switching to a
/// `Vec<OptionEntry>` indexed by each option's position in `OPTION_DEFS`.
/// This eliminates hashing entirely and gives cache-friendly access.

use std::collections::HashMap;

/// Typed value for a simulator option.
#[derive(Debug, Clone)]
pub enum OptionValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

impl std::fmt::Display for OptionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionValue::Int(v) => write!(f, "{}", v),
            OptionValue::Float(v) => write!(f, "{}", v),
            OptionValue::Str(v) => write!(f, "{}", v),
            OptionValue::Bool(v) => write!(f, "{}", v),
        }
    }
}

/// Range constraint for validating option values.
#[derive(Debug, Clone)]
pub enum OptionRange {
    /// No constraint.
    None,
    /// Exclusive range for integers: (min, max).
    IntRange(i64, i64),
    /// Exclusive range for floats: (min, max).
    FloatRange(f64, f64),
    /// Enumerated set of allowed string values.
    StringEnum(Vec<&'static str>),
}

/// Static definition of one simulator option.
#[derive(Debug, Clone)]
pub struct OptionDef {
    pub name: &'static str,
    pub description: &'static str,
    pub default: OptionValue,
    pub range: OptionRange,
}

/// All known simulator options. To add a new option, add one entry here.
const OPTION_DEFS: &[OptionDef] = &[
    OptionDef {
        name: "abstol",
        description: "Absolute current tolerance",
        default: OptionValue::Float(1e-12),
        range: OptionRange::FloatRange(0.0, 1e-3),
    },
    OptionDef {
        name: "reltol",
        description: "Relative tolerance",
        default: OptionValue::Float(1e-3),
        range: OptionRange::FloatRange(0.0, 1.0),
    },
    OptionDef {
        name: "vntol",
        description: "Voltage node tolerance",
        default: OptionValue::Float(1e-6),
        range: OptionRange::FloatRange(0.0, 1.0),
    },
    OptionDef {
        name: "gmin",
        description: "Minimum conductance",
        default: OptionValue::Float(1e-12),
        range: OptionRange::FloatRange(0.0, 1e-3),
    },
    OptionDef {
        name: "itl1",
        description: "DC iteration limit",
        default: OptionValue::Int(100),
        range: OptionRange::IntRange(1, 10000),
    },
    OptionDef {
        name: "itl4",
        description: "Transient iteration limit",
        default: OptionValue::Int(50),
        range: OptionRange::IntRange(1, 10000),
    },
    OptionDef {
        name: "temp",
        description: "Temperature in Celsius",
        default: OptionValue::Float(27.0),
        range: OptionRange::FloatRange(-273.15, 1000.0),
    },
    OptionDef {
        name: "tnom",
        description: "Nominal temperature in Celsius",
        default: OptionValue::Float(27.0),
        range: OptionRange::FloatRange(-273.15, 1000.0),
    },
];

/// A stored option entry with its current value and whether the user set it.
#[derive(Debug, Clone)]
struct OptionEntry {
    value: OptionValue,
    is_set: bool,
}

/// Container for all simulator options.
///
/// Constructed with defaults from `OPTION_DEFS`. Call `set()` to apply
/// user-specified values from `.option` netlist directives.
#[derive(Debug, Clone)]
pub struct SimOptions {
    entries: HashMap<String, OptionEntry>,
}

impl SimOptions {
    /// Create a new `SimOptions` populated with defaults from `OPTION_DEFS`.
    pub fn new() -> Self {
        let mut entries = HashMap::new();
        for def in OPTION_DEFS {
            entries.insert(
                def.name.to_string(),
                OptionEntry {
                    value: def.default.clone(),
                    is_set: false,
                },
            );
        }
        Self { entries }
    }

    /// Set an option by name from a raw string value.
    ///
    /// Parses the value according to the option's type, validates against
    /// its range, and stores it. Warns on unknown options, parse errors,
    /// and out-of-range values. If the option was already set, warns about
    /// the redefinition.
    pub fn set(&mut self, key: &str, raw_value: &str) {
        let key_lower = key.to_ascii_lowercase();

        // Find the definition for this option
        let def = match OPTION_DEFS.iter().find(|d| d.name == key_lower) {
            Some(d) => d,
            None => {
                eprintln!("warning: unknown option '{}' ignored", key);
                return;
            }
        };

        // Parse value according to expected type
        let parsed = match &def.default {
            OptionValue::Float(_) => {
                if let Some(v) = parse_option_float(raw_value) {
                    OptionValue::Float(v)
                } else {
                    eprintln!(
                        "warning: option '{}' value '{}' is not a valid number, ignored",
                        key_lower, raw_value
                    );
                    return;
                }
            }
            OptionValue::Int(_) => {
                if let Some(v) = parse_option_int(raw_value) {
                    OptionValue::Int(v)
                } else {
                    eprintln!(
                        "warning: option '{}' value '{}' is not a valid integer, ignored",
                        key_lower, raw_value
                    );
                    return;
                }
            }
            OptionValue::Str(_) => OptionValue::Str(raw_value.to_string()),
            OptionValue::Bool(_) => {
                let v = matches!(
                    raw_value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                );
                OptionValue::Bool(v)
            }
        };

        // Validate range
        if !validate_range(&parsed, &def.range) {
            eprintln!(
                "warning: option '{}' value {} out of range ({}), using default {}",
                key_lower,
                parsed,
                format_range(&def.range),
                def.default
            );
            return;
        }

        // Check for redefinition
        if let Some(entry) = self.entries.get(&key_lower) {
            if entry.is_set {
                eprintln!(
                    "warning: option '{}' redefined (was {}), using new value {}",
                    key_lower, entry.value, parsed
                );
            }
        }

        self.entries.insert(
            key_lower,
            OptionEntry {
                value: parsed,
                is_set: true,
            },
        );
    }

    /// Get a float option value (returns default if not found).
    pub fn get_float(&self, key: &str) -> f64 {
        match self.entries.get(key) {
            Some(entry) => match &entry.value {
                OptionValue::Float(v) => *v,
                _ => 0.0,
            },
            None => 0.0,
        }
    }

    /// Get an integer option value (returns default if not found).
    pub fn get_int(&self, key: &str) -> i64 {
        match self.entries.get(key) {
            Some(entry) => match &entry.value {
                OptionValue::Int(v) => *v,
                _ => 0,
            },
            None => 0,
        }
    }

    /// Get a string option value (returns default if not found).
    pub fn get_string(&self, key: &str) -> &str {
        match self.entries.get(key) {
            Some(entry) => match &entry.value {
                OptionValue::Str(v) => v.as_str(),
                _ => "",
            },
            None => "",
        }
    }

    /// Get a boolean option value (returns default if not found).
    pub fn get_bool(&self, key: &str) -> bool {
        match self.entries.get(key) {
            Some(entry) => match &entry.value {
                OptionValue::Bool(v) => *v,
                _ => false,
            },
            None => false,
        }
    }

    /// Check whether the user explicitly set this option.
    pub fn is_set(&self, key: &str) -> bool {
        self.entries.get(key).map_or(false, |e| e.is_set)
    }

    /// Print all user-set options. Prints nothing if no options were set.
    pub fn print_user_options(&self) {
        let mut user_set: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_set)
            .collect();

        if user_set.is_empty() {
            return;
        }

        user_set.sort_by_key(|(name, _)| (*name).clone());
        println!("options:");
        for (name, entry) in &user_set {
            println!("  {} = {}  (user-set)", name, entry.value);
        }
    }
}

fn validate_range(value: &OptionValue, range: &OptionRange) -> bool {
    match (value, range) {
        (_, OptionRange::None) => true,
        (OptionValue::Float(v), OptionRange::FloatRange(min, max)) => *v > *min && *v < *max,
        (OptionValue::Int(v), OptionRange::IntRange(min, max)) => *v >= *min && *v <= *max,
        (OptionValue::Str(v), OptionRange::StringEnum(allowed)) => {
            allowed.iter().any(|a| a.eq_ignore_ascii_case(v))
        }
        _ => true,
    }
}

fn format_range(range: &OptionRange) -> String {
    match range {
        OptionRange::None => "no range".to_string(),
        OptionRange::IntRange(min, max) => format!("{} to {}", min, max),
        OptionRange::FloatRange(min, max) => format!("{} to {}", min, max),
        OptionRange::StringEnum(values) => format!("one of: {}", values.join(", ")),
    }
}

fn parse_option_float(s: &str) -> Option<f64> {
    // Try SPICE suffix first, then plain parse
    crate::netlist::parse_number_with_suffix(s).or_else(|| s.parse().ok())
}

fn parse_option_int(s: &str) -> Option<i64> {
    s.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let opts = SimOptions::new();
        assert!((opts.get_float("abstol") - 1e-12).abs() < 1e-20);
        assert!((opts.get_float("reltol") - 1e-3).abs() < 1e-10);
        assert!((opts.get_float("vntol") - 1e-6).abs() < 1e-13);
        assert!((opts.get_float("gmin") - 1e-12).abs() < 1e-20);
        assert_eq!(opts.get_int("itl1"), 100);
        assert_eq!(opts.get_int("itl4"), 50);
        assert!((opts.get_float("temp") - 27.0).abs() < 1e-10);
        assert!((opts.get_float("tnom") - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_float() {
        let mut opts = SimOptions::new();
        opts.set("abstol", "1e-15");
        assert!((opts.get_float("abstol") - 1e-15).abs() < 1e-23);
        assert!(opts.is_set("abstol"));
    }

    #[test]
    fn test_set_int() {
        let mut opts = SimOptions::new();
        opts.set("itl1", "200");
        assert_eq!(opts.get_int("itl1"), 200);
        assert!(opts.is_set("itl1"));
    }

    #[test]
    fn test_is_set_default() {
        let opts = SimOptions::new();
        assert!(!opts.is_set("abstol"));
        assert!(!opts.is_set("itl1"));
    }

    #[test]
    fn test_case_insensitive() {
        let mut opts = SimOptions::new();
        opts.set("ABSTOL", "1e-14");
        assert!((opts.get_float("abstol") - 1e-14).abs() < 1e-22);
        assert!(opts.is_set("abstol"));
    }

    #[test]
    fn test_out_of_range_float() {
        let mut opts = SimOptions::new();
        // abstol range is (0, 1e-3), so 1.0 is out of range
        opts.set("abstol", "1.0");
        // Should keep default
        assert!((opts.get_float("abstol") - 1e-12).abs() < 1e-20);
        assert!(!opts.is_set("abstol"));
    }

    #[test]
    fn test_out_of_range_int() {
        let mut opts = SimOptions::new();
        // itl1 range is (1, 10000), so 0 is out of range
        opts.set("itl1", "0");
        // Should keep default
        assert_eq!(opts.get_int("itl1"), 100);
        assert!(!opts.is_set("itl1"));
    }

    #[test]
    fn test_unknown_option() {
        let mut opts = SimOptions::new();
        opts.set("nonexistent", "42");
        // Should not crash, just warn
        assert!(!opts.is_set("nonexistent"));
    }

    #[test]
    fn test_duplicate_set() {
        let mut opts = SimOptions::new();
        opts.set("abstol", "1e-15");
        assert!((opts.get_float("abstol") - 1e-15).abs() < 1e-23);
        // Set again - should warn but accept new value
        opts.set("abstol", "1e-14");
        assert!((opts.get_float("abstol") - 1e-14).abs() < 1e-22);
    }

    #[test]
    fn test_spice_suffix() {
        let mut opts = SimOptions::new();
        opts.set("abstol", "1p");
        assert!((opts.get_float("abstol") - 1e-12).abs() < 1e-20);
    }

    #[test]
    fn test_temp_range() {
        let mut opts = SimOptions::new();
        opts.set("temp", "85");
        assert!((opts.get_float("temp") - 85.0).abs() < 1e-10);

        // Absolute zero boundary
        opts.set("temp", "-300");
        // Out of range, should keep previous
        assert!((opts.get_float("temp") - 85.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_value() {
        let mut opts = SimOptions::new();
        opts.set("abstol", "abc");
        // Should keep default
        assert!((opts.get_float("abstol") - 1e-12).abs() < 1e-20);
        assert!(!opts.is_set("abstol"));
    }
}
