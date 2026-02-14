//! Complex MNA (Modified Nodal Analysis) matrix builder for AC analysis.
//!
//! This module provides data structures for building complex admittance matrices
//! Y(jω) used in AC small-signal frequency-domain analysis.

use num_complex::Complex64;
use std::collections::HashMap;

/// Sparse matrix builder for complex numbers in CSC (Compressed Sparse Column) format.
#[derive(Debug, Clone)]
pub struct ComplexSparseBuilder {
    pub n: usize,
    pub col_entries: Vec<Vec<(usize, Complex64)>>,
}

impl ComplexSparseBuilder {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            col_entries: vec![Vec::new(); n],
        }
    }

    pub fn insert(&mut self, col: usize, row: usize, value: Complex64) {
        if col >= self.n {
            return;
        }
        self.col_entries[col].push((row, value));
    }

    pub fn resize(&mut self, new_n: usize) {
        if new_n <= self.n {
            return;
        }
        self.col_entries.resize_with(new_n, Vec::new);
        self.n = new_n;
    }

    /// Finalize the sparse matrix into CSC format.
    /// Returns (column pointers, row indices, values).
    pub fn finalize(&mut self) -> (Vec<i64>, Vec<i64>, Vec<Complex64>) {
        let mut ap = Vec::with_capacity(self.n + 1);
        let mut ai = Vec::new();
        let mut ax = Vec::new();

        let mut nnz = 0i64;
        ap.push(0);
        for col in &mut self.col_entries {
            col.sort_by_key(|(row, _)| *row);
            for (row, value) in col.iter() {
                ai.push(*row as i64);
                ax.push(*value);
                nnz += 1;
            }
            ap.push(nnz);
        }

        (ap, ai, ax)
    }
}

/// Auxiliary variable table for complex MNA.
#[derive(Debug, Clone)]
pub struct ComplexAuxVarTable {
    pub name_to_id: HashMap<String, usize>,
    pub id_to_name: Vec<String>,
}

impl ComplexAuxVarTable {
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
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

/// Context for stamping complex admittances into the MNA matrix.
#[derive(Debug)]
pub struct ComplexStampContext<'a> {
    pub builder: &'a mut ComplexSparseBuilder,
    pub rhs: &'a mut Vec<Complex64>,
    pub aux: &'a mut ComplexAuxVarTable,
    pub node_count: usize,
    /// Angular frequency ω = 2πf
    pub omega: f64,
}

impl<'a> ComplexStampContext<'a> {
    /// Add a complex admittance value to the matrix at position (i, j).
    pub fn add(&mut self, i: usize, j: usize, value: Complex64) {
        self.builder.insert(j, i, value);
    }

    /// Add a real admittance value to the matrix at position (i, j).
    pub fn add_real(&mut self, i: usize, j: usize, value: f64) {
        self.add(i, j, Complex64::new(value, 0.0));
    }

    /// Add a purely imaginary admittance value to the matrix at position (i, j).
    pub fn add_imag(&mut self, i: usize, j: usize, value: f64) {
        self.add(i, j, Complex64::new(0.0, value));
    }

    /// Add a complex value to the right-hand side vector.
    pub fn add_rhs(&mut self, i: usize, value: Complex64) {
        if let Some(entry) = self.rhs.get_mut(i) {
            *entry += value;
        }
    }

    /// Add a real value to the right-hand side vector.
    pub fn add_rhs_real(&mut self, i: usize, value: f64) {
        self.add_rhs(i, Complex64::new(value, 0.0));
    }

    /// Allocate an auxiliary variable (for voltage sources, inductors, etc.).
    pub fn allocate_aux(&mut self, name: &str) -> usize {
        let (aux_id, is_new) = self.aux.allocate_with_flag(name);
        let index = self.node_count + aux_id;
        if is_new {
            self.builder.resize(self.node_count + self.aux.id_to_name.len());
            self.rhs.resize(self.builder.n, Complex64::new(0.0, 0.0));
        }
        index
    }
}

/// High-level builder for complex MNA matrices.
#[derive(Debug)]
pub struct ComplexMnaBuilder {
    pub node_count: usize,
    pub size: usize,
    pub rhs: Vec<Complex64>,
    pub builder: ComplexSparseBuilder,
    pub aux: ComplexAuxVarTable,
}

impl ComplexMnaBuilder {
    pub fn new(node_count: usize) -> Self {
        let size = node_count;
        Self {
            node_count,
            size,
            rhs: vec![Complex64::new(0.0, 0.0); size],
            builder: ComplexSparseBuilder::new(size),
            aux: ComplexAuxVarTable::new(),
        }
    }

    /// Get a stamp context with the given angular frequency.
    pub fn context(&mut self, omega: f64) -> ComplexStampContext<'_> {
        ComplexStampContext {
            builder: &mut self.builder,
            rhs: &mut self.rhs,
            aux: &mut self.aux,
            node_count: self.node_count,
            omega,
        }
    }
}
