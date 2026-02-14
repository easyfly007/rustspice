#[derive(Debug, Clone)]
pub struct MnaSystem {
    pub size: usize,
}

pub fn debug_dump_mna(system: &MnaSystem) {
    println!("mna: size={}", system.size);
}

#[derive(Debug, Clone)]
pub struct AuxVarTable {
    pub name_to_id: std::collections::HashMap<String, usize>,
    pub id_to_name: Vec<String>,
}

impl AuxVarTable {
    pub fn new() -> Self {
        Self {
            name_to_id: std::collections::HashMap::new(),
            id_to_name: Vec::new(),
        }
    }

    pub fn allocate(&mut self, name: &str) -> usize {
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }
        let id = self.id_to_name.len();
        self.name_to_id.insert(name.to_string(), id);
        self.id_to_name.push(name.to_string());
        id
    }

    pub fn allocate_with_flag(&mut self, name: &str) -> (usize, bool) {
        if let Some(id) = self.name_to_id.get(name) {
            return (*id, false);
        }
        let id = self.id_to_name.len();
        self.name_to_id.insert(name.to_string(), id);
        self.id_to_name.push(name.to_string());
        (id, true)
    }
}

use std::collections::HashMap;

/// Sparse matrix builder with pattern caching for efficient repeated assembly.
///
/// # Operating Modes
///
/// The builder operates in two modes:
///
/// 1. **Dynamic mode** (default): Entries are accumulated in vectors. Supports
///    arbitrary insertions but requires O(nnz log nnz) finalization.
///
/// 2. **Frozen mode**: After calling `freeze_pattern()`, the sparsity structure
///    is locked and values can be updated in O(1) via `update()`.
///
/// # Performance Characteristics
///
/// | Operation | Dynamic Mode | Frozen Mode |
/// |-----------|--------------|-------------|
/// | insert()  | O(1) amortized | O(1) with index_map |
/// | update()  | N/A | O(1) |
/// | clear_values() | O(nnz) | O(nnz) |
/// | finalize() | O(nnz log nnz) | O(nnz) |
///
/// # Example
///
/// ```ignore
/// let mut builder = SparseBuilder::new(3);
///
/// // First assembly (dynamic mode)
/// builder.insert(0, 0, 1.0);
/// builder.insert(0, 1, 2.0);
/// builder.insert(1, 1, 3.0);
/// let (ap, ai, ax) = builder.finalize_merged();
///
/// // Freeze pattern for repeated solves
/// builder.freeze_pattern();
///
/// // Fast updates (frozen mode)
/// builder.clear_values();
/// builder.update(0, 0, 1.5);  // O(1) lookup
/// builder.update(0, 1, 2.5);
/// let ax_only = builder.get_values();  // Pattern unchanged
/// ```
#[derive(Debug, Clone)]
pub struct SparseBuilder {
    /// Matrix dimension
    pub n: usize,

    /// Column-wise entries: col_entries[col] = [(row, value), ...]
    /// Used in dynamic mode for accumulating entries.
    pub col_entries: Vec<Vec<(usize, f64)>>,

    /// Whether the pattern is frozen (index_map is valid)
    frozen: bool,

    /// Index map for O(1) lookups: index_map[col][(row)] = index in values array
    /// Only valid when frozen = true.
    index_map: Vec<HashMap<usize, usize>>,

    /// CSC structure (only valid after freeze_pattern or finalize_merged)
    cached_ap: Vec<i64>,
    cached_ai: Vec<i64>,

    /// Values array for frozen mode
    cached_ax: Vec<f64>,

    /// Statistics
    stats: SparseBuilderStats,
}

/// Statistics for monitoring SparseBuilder performance.
#[derive(Debug, Clone, Default)]
pub struct SparseBuilderStats {
    /// Number of insert() calls
    pub inserts: usize,
    /// Number of update() calls (frozen mode)
    pub updates: usize,
    /// Number of times pattern was frozen
    pub freezes: usize,
    /// Number of times pattern was unfrozen
    pub unfreezes: usize,
    /// Number of duplicate entries merged in finalize
    pub duplicates_merged: usize,
    /// Total non-zeros after merging
    pub nnz: usize,
}

impl SparseBuilder {
    /// Create a new sparse builder for an n×n matrix.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            col_entries: vec![Vec::new(); n],
            frozen: false,
            index_map: Vec::new(),
            cached_ap: Vec::new(),
            cached_ai: Vec::new(),
            cached_ax: Vec::new(),
            stats: SparseBuilderStats::default(),
        }
    }

    /// Insert a value at (row, col). Values at the same position are accumulated.
    ///
    /// In dynamic mode: appends to col_entries (merged later in finalize).
    /// In frozen mode: uses index_map for O(1) accumulation.
    pub fn insert(&mut self, col: usize, row: usize, value: f64) {
        if col >= self.n || row >= self.n {
            return;
        }

        self.stats.inserts += 1;

        if self.frozen {
            // Frozen mode: use index map for O(1) update
            if let Some(idx) = self.index_map.get(col).and_then(|m| m.get(&row)) {
                self.cached_ax[*idx] += value;
            }
            // If position not in pattern, silently ignore (pattern is fixed)
        } else {
            // Dynamic mode: append to col_entries
            self.col_entries[col].push((row, value));
        }
    }

    /// Update a value at (row, col) in frozen mode. Replaces (not accumulates).
    ///
    /// This is O(1) when pattern is frozen.
    /// Returns true if the position exists in the pattern.
    pub fn update(&mut self, col: usize, row: usize, value: f64) -> bool {
        if !self.frozen || col >= self.n {
            return false;
        }

        self.stats.updates += 1;

        if let Some(idx) = self.index_map.get(col).and_then(|m| m.get(&row)) {
            self.cached_ax[*idx] = value;
            true
        } else {
            false
        }
    }

    /// Accumulate a value at (row, col) in frozen mode.
    ///
    /// This is O(1) when pattern is frozen.
    /// Returns true if the position exists in the pattern.
    pub fn accumulate(&mut self, col: usize, row: usize, value: f64) -> bool {
        if !self.frozen || col >= self.n {
            return false;
        }

        if let Some(idx) = self.index_map.get(col).and_then(|m| m.get(&row)) {
            self.cached_ax[*idx] += value;
            true
        } else {
            false
        }
    }

    /// Clear all values to zero, preserving the sparsity pattern.
    pub fn clear_values(&mut self) {
        if self.frozen {
            // Fast path: just zero the values array
            self.cached_ax.fill(0.0);
        } else {
            // Dynamic mode: zero entries in place
            for col in &mut self.col_entries {
                for entry in col.iter_mut() {
                    entry.1 = 0.0;
                }
            }
        }
    }

    /// Clear all entries and reset to empty matrix (unfreezes pattern).
    pub fn clear_all(&mut self) {
        for col in &mut self.col_entries {
            col.clear();
        }
        self.frozen = false;
        self.index_map.clear();
        self.cached_ap.clear();
        self.cached_ai.clear();
        self.cached_ax.clear();
        self.stats.unfreezes += 1;
    }

    /// Resize the matrix dimension. Unfreezes pattern if size changes.
    pub fn resize(&mut self, new_n: usize) {
        if new_n <= self.n {
            return;
        }

        // Unfreeze if pattern would change
        if self.frozen {
            self.frozen = false;
            self.index_map.clear();
            self.cached_ap.clear();
            self.cached_ai.clear();
            self.cached_ax.clear();
            self.stats.unfreezes += 1;
        }

        self.col_entries.resize_with(new_n, Vec::new);
        self.n = new_n;
    }

    /// Check if the pattern is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Get the number of non-zeros (after merging duplicates).
    pub fn nnz(&self) -> usize {
        if self.frozen {
            self.cached_ax.len()
        } else {
            self.col_entries.iter().map(|c| c.len()).sum()
        }
    }

    /// Finalize and return CSC arrays, merging duplicate entries.
    ///
    /// Duplicate entries at the same (row, col) are summed together.
    /// This is more correct for circuit simulation where stamps accumulate.
    pub fn finalize_merged(&mut self) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        if self.frozen {
            // Already finalized, return cached values
            return (
                self.cached_ap.clone(),
                self.cached_ai.clone(),
                self.cached_ax.clone(),
            );
        }

        let mut ap = Vec::with_capacity(self.n + 1);
        let mut ai = Vec::new();
        let mut ax = Vec::new();
        let mut duplicates = 0;

        ap.push(0);

        for col in &mut self.col_entries {
            // Sort by row index
            col.sort_by_key(|(row, _)| *row);

            // Merge duplicates
            let mut last_row: Option<usize> = None;
            for (row, value) in col.iter() {
                if Some(*row) == last_row {
                    // Duplicate: accumulate into last entry
                    if let Some(last) = ax.last_mut() {
                        *last += value;
                    }
                    duplicates += 1;
                } else {
                    // New entry
                    ai.push(*row as i64);
                    ax.push(*value);
                    last_row = Some(*row);
                }
            }

            ap.push(ai.len() as i64);
        }

        self.stats.duplicates_merged += duplicates;
        self.stats.nnz = ax.len();

        (ap, ai, ax)
    }

    /// Finalize without merging (legacy behavior). Duplicates remain separate.
    pub fn finalize(&mut self) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        if self.frozen {
            return (
                self.cached_ap.clone(),
                self.cached_ai.clone(),
                self.cached_ax.clone(),
            );
        }

        let mut ap = Vec::with_capacity(self.n + 1);
        let mut ai = Vec::new();
        let mut ax = Vec::new();

        let mut nnz = 0;
        ap.push(0);
        for col in &mut self.col_entries {
            col.sort_by_key(|(row, _)| *row);
            for (row, value) in col.iter() {
                ai.push(*row as i64);
                ax.push(*value);
                nnz += 1;
            }
            ap.push(nnz as i64);
        }

        self.stats.nnz = nnz;
        (ap, ai, ax)
    }

    /// Freeze the sparsity pattern for efficient repeated assembly.
    ///
    /// After freezing:
    /// - `insert()` uses O(1) index lookup (but only for existing positions)
    /// - `update()` and `accumulate()` become available
    /// - `clear_values()` is O(nnz) but preserves structure
    /// - `finalize()` returns cached arrays in O(1)
    ///
    /// Call `unfreeze()` or `clear_all()` to modify the pattern again.
    pub fn freeze_pattern(&mut self) {
        if self.frozen {
            return;
        }

        // Finalize with merging to get clean CSC structure
        let (ap, ai, ax) = self.finalize_merged();

        // Build index map for O(1) lookups
        let mut index_map = vec![HashMap::new(); self.n];

        for col in 0..self.n {
            let start = ap[col] as usize;
            let end = ap[col + 1] as usize;

            for (idx, &row) in ai[start..end].iter().enumerate() {
                index_map[col].insert(row as usize, start + idx);
            }
        }

        self.cached_ap = ap;
        self.cached_ai = ai;
        self.cached_ax = ax;
        self.index_map = index_map;
        self.frozen = true;
        self.stats.freezes += 1;
    }

    /// Unfreeze the pattern to allow structural modifications.
    ///
    /// This converts the frozen CSC back to dynamic col_entries.
    pub fn unfreeze(&mut self) {
        if !self.frozen {
            return;
        }

        // Rebuild col_entries from CSC
        self.col_entries = vec![Vec::new(); self.n];

        for col in 0..self.n {
            let start = self.cached_ap[col] as usize;
            let end = self.cached_ap[col + 1] as usize;

            for i in start..end {
                let row = self.cached_ai[i] as usize;
                let value = self.cached_ax[i];
                self.col_entries[col].push((row, value));
            }
        }

        self.frozen = false;
        self.index_map.clear();
        self.stats.unfreezes += 1;
    }

    /// Get the cached values array (only valid when frozen).
    pub fn get_values(&self) -> Option<&[f64]> {
        if self.frozen {
            Some(&self.cached_ax)
        } else {
            None
        }
    }

    /// Get mutable access to cached values (only valid when frozen).
    pub fn get_values_mut(&mut self) -> Option<&mut [f64]> {
        if self.frozen {
            Some(&mut self.cached_ax)
        } else {
            None
        }
    }

    /// Get the cached CSC structure (only valid when frozen).
    pub fn get_csc(&self) -> Option<(&[i64], &[i64], &[f64])> {
        if self.frozen {
            Some((&self.cached_ap, &self.cached_ai, &self.cached_ax))
        } else {
            None
        }
    }

    /// Get builder statistics.
    pub fn stats(&self) -> &SparseBuilderStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SparseBuilderStats::default();
    }
}

#[derive(Debug)]
pub struct StampContext<'a> {
    pub builder: &'a mut SparseBuilder,
    pub rhs: &'a mut Vec<f64>,
    pub aux: &'a mut AuxVarTable,
    pub node_count: usize,
    pub gmin: f64,
    pub source_scale: f64,
}

impl<'a> StampContext<'a> {
    pub fn add(&mut self, i: usize, j: usize, value: f64) {
        self.builder.insert(j, i, value);
    }

    pub fn add_rhs(&mut self, i: usize, value: f64) {
        if let Some(entry) = self.rhs.get_mut(i) {
            *entry += value;
        }
    }

    pub fn allocate_aux(&mut self, name: &str) -> usize {
        let (aux_id, is_new) = self.aux.allocate_with_flag(name);
        let index = self.node_count + aux_id;
        if is_new {
            self.builder.resize(self.node_count + self.aux.id_to_name.len());
            self.rhs.resize(self.builder.n, 0.0);
        }
        index
    }
}

#[derive(Debug)]
pub struct MnaBuilder {
    pub node_count: usize,
    pub size: usize,
    pub rhs: Vec<f64>,
    pub builder: SparseBuilder,
    pub aux: AuxVarTable,
}

impl MnaBuilder {
    pub fn new(node_count: usize) -> Self {
        let size = node_count;
        Self {
            node_count,
            size,
            rhs: vec![0.0; size],
            builder: SparseBuilder::new(size),
            aux: AuxVarTable::new(),
        }
    }

    pub fn context(&mut self) -> StampContext<'_> {
        StampContext {
            builder: &mut self.builder,
            rhs: &mut self.rhs,
            aux: &mut self.aux,
            node_count: self.node_count,
            gmin: 0.0,
            source_scale: 1.0,
        }
    }

    pub fn context_with(&mut self, gmin: f64, source_scale: f64) -> StampContext<'_> {
        StampContext {
            builder: &mut self.builder,
            rhs: &mut self.rhs,
            aux: &mut self.aux,
            node_count: self.node_count,
            gmin,
            source_scale,
        }
    }
}
