use super::{McAlgorithm, SweepResult};
use crate::lattice::Lattice;
use rand::Rng;

/// Sweep strategy, selected once at construction based on which Hamiltonian
/// terms are active (nonzero). Named by active terms: J1, J2, J3, H.
///
/// Each variant has a dedicated sweep method with direct lattice method calls
/// (no function pointers). LLVM monomorphizes per lattice type + unrolls
/// the iter().sum() neighbor loops, producing equivalent code to hand-unrolled
/// sums. Verified: 268M updates/sec with iter().sum() vs 272M hand-unrolled.
#[derive(Clone, Copy, Debug)]
enum SweepStrategy {
    J1,
    J2,
    H,
    J3,
    J1H,
    J2H,
    J3H,
    J1J2,
    J1J3,
    J2J3,
    J1J2H,
    J1J3H,
    J2J3H,
    J1J2J3,
    J1J2J3H,
}

/// Metropolis single-spin-flip algorithm.
///
/// Uses a sequential scan (systematic Metropolis) for cache locality.
/// The sweep strategy is selected once at construction and dispatched
/// via a single match per sweep call (not per flip).
///
/// Tables are `Vec<f64>` sized by coordination number at construction,
/// supporting any lattice geometry. The Vec data pointer is cached in L1
/// after the first site access — no measurable overhead vs fixed arrays.
pub struct Metropolis {
    strategy: SweepStrategy,
    /// Coordination numbers (from lattice).
    z_nn: usize,
    z_nnn: usize,
    z_tnn: usize,

    // ── Lookup tables ────────────────────────────────────────────────
    // Each table stores min(1, exp(-beta * dE)) for all possible dE values.
    // Sized by coordination number: z neighbors → z+1 possible sum values.

    /// Single-coupling sign-branch table: ceil(z/2) entries for positive dE only.
    table_sign: Vec<f64>,
    cached_key_sign: f64,

    /// Field-only table: 2 entries [spin=-1, spin=+1].
    table_field: [f64; 2],
    cached_key_field: f64,

    /// Coupling+field branchless table: 2*(z+1) entries [spin × sum].
    table_cf: Vec<f64>,
    cached_key_cf: (f64, f64, f64),

    /// Two-coupling spin-symmetry table: (z_a+1)*(z_b+1) entries.
    table_2c: Vec<f64>,
    cached_key_2c: (f64, f64, f64),

    /// Two-coupling+field branchless table: 2*(z_a+1)*(z_b+1) entries.
    table_2cf: Vec<f64>,
    cached_key_2cf: (f64, f64, f64, f64),

    /// Three-coupling spin-symmetry table: (z_nn+1)*(z_nnn+1)*(z_tnn+1) entries.
    table_3c: Vec<f64>,
    cached_key_3c: (f64, f64, f64, f64),

    /// Three-coupling+field branchless table: 2*(z_nn+1)*(z_nnn+1)*(z_tnn+1).
    table_3cf: Vec<f64>,
    cached_key_3cf: (f64, f64, f64, f64, f64),
}

impl Metropolis {
    pub fn new(j1: f64, j2: f64, j3: f64, h: f64, z_nn: usize, z_nnn: usize, z_tnn: usize) -> Self {
        let strategy = match (j1 != 0.0, j2 != 0.0, j3 != 0.0, h != 0.0) {
            (true, true, true, true) => SweepStrategy::J1J2J3H,
            (true, true, true, false) => SweepStrategy::J1J2J3,
            (true, true, false, true) => SweepStrategy::J1J2H,
            (true, true, false, false) => SweepStrategy::J1J2,
            (true, false, true, true) => SweepStrategy::J1J3H,
            (true, false, true, false) => SweepStrategy::J1J3,
            (true, false, false, true) => SweepStrategy::J1H,
            (true, false, false, false) => SweepStrategy::J1,
            (false, true, true, true) => SweepStrategy::J2J3H,
            (false, true, true, false) => SweepStrategy::J2J3,
            (false, true, false, true) => SweepStrategy::J2H,
            (false, true, false, false) => SweepStrategy::J2,
            (false, false, true, true) => SweepStrategy::J3H,
            (false, false, true, false) => SweepStrategy::J3,
            (false, false, false, true) => SweepStrategy::H,
            (false, false, false, false) => SweepStrategy::J1, // degenerate
        };

        // Pre-allocate tables with correct sizes for this lattice's coordination.
        // Max single-coupling sign-branch entries: ceil(z/2) for the largest z used.
        let max_z = z_nn.max(z_nnn).max(z_tnn);
        let sign_size = (max_z + 1) / 2; // ceil(z/2)

        // Coupling+field: 2 * (z+1) for the active single coupling
        let cf_size = 2 * (max_z + 1);

        // Two-coupling sizes depend on which pair. Allocate for the largest.
        let twoc_size = (z_nn + 1).max(z_nnn + 1).max(z_tnn + 1);
        let table_2c_size = twoc_size * twoc_size;
        let table_2cf_size = 2 * twoc_size * twoc_size;

        // Three-coupling
        let table_3c_size = (z_nn + 1) * (z_nnn + 1) * (z_tnn + 1);
        let table_3cf_size = 2 * table_3c_size;

        Self {
            strategy,
            z_nn,
            z_nnn,
            z_tnn,
            table_sign: vec![0.0; sign_size],
            cached_key_sign: f64::NAN,
            table_field: [0.0; 2],
            cached_key_field: f64::NAN,
            table_cf: vec![0.0; cf_size],
            cached_key_cf: (f64::NAN, f64::NAN, f64::NAN),
            table_2c: vec![0.0; table_2c_size],
            cached_key_2c: (f64::NAN, f64::NAN, f64::NAN),
            table_2cf: vec![0.0; table_2cf_size],
            cached_key_2cf: (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
            table_3c: vec![0.0; table_3c_size],
            cached_key_3c: (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
            table_3cf: vec![0.0; table_3cf_size],
            cached_key_3cf: (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        }
    }

    // ── Table ensure methods ──────────────────────────────────────────
    //
    // General index formula for coordination z:
    //   sum ∈ {-z, -z+2, ..., z} → (z+1) possible values
    //   sum_idx = (sum + z) / 2  → range [0, z]
    //
    // Sign-branch tables store only positive-dE entries:
    //   half_de = spin * sum (positive in the else branch)
    //   table_idx = (half_de - 1) / 2  → range [0, ceil(z/2)-1]
    //   Works for both even and odd z.

    /// Single-coupling sign-branch table: entries for positive dE only.
    #[inline]
    fn ensure_table_sign(&mut self, beta: f64, coupling: f64, z: usize) {
        let key = beta * coupling;
        if key != self.cached_key_sign {
            let n_entries = (z + 1) / 2;
            for i in 0..n_entries {
                // half_de values: 1,3,5,... (odd z) or 2,4,6,... (even z)
                let half_de = if z % 2 == 0 { 2 * (i + 1) } else { 2 * i + 1 };
                let de = 2.0 * half_de as f64;
                self.table_sign[i] = (-beta * coupling * de).exp();
            }
            self.cached_key_sign = key;
        }
    }

    /// Field-only table: 2 entries [spin=-1, spin=+1].
    #[inline]
    fn ensure_table_field(&mut self, beta: f64, h: f64) {
        let key = beta * h;
        if key != self.cached_key_field {
            self.table_field[0] = (2.0 * key).exp().min(1.0);  // spin=-1
            self.table_field[1] = (-2.0 * key).exp().min(1.0); // spin=+1
            self.cached_key_field = key;
        }
    }

    /// Coupling+field branchless table: 2*(z+1) entries.
    #[inline]
    fn ensure_table_cf(&mut self, beta: f64, coupling: f64, h: f64, z: usize) {
        let key = (beta, coupling, h);
        if key != self.cached_key_cf {
            let zp1 = z + 1;
            for (spin_idx, &spin) in [-1.0_f64, 1.0].iter().enumerate() {
                for i in 0..zp1 {
                    let sum = (i as f64 * 2.0) - z as f64;
                    let de = 2.0 * spin * (coupling * sum + h);
                    self.table_cf[spin_idx * zp1 + i] = (-beta * de).exp().min(1.0);
                }
            }
            self.cached_key_cf = key;
        }
    }

    /// Two-coupling spin-symmetry table: (z_a+1)*(z_b+1) entries.
    /// Precomputed for spin=+1; spin=-1 mirrors via table[max_idx - idx].
    #[inline]
    fn ensure_table_2c(&mut self, beta: f64, ca: f64, cb: f64, za: usize, zb: usize) {
        let key = (beta, ca, cb);
        if key != self.cached_key_2c {
            let zap1 = za + 1;
            let zbp1 = zb + 1;
            for ia in 0..zap1 {
                let sum_a = (ia as f64 * 2.0) - za as f64;
                for ib in 0..zbp1 {
                    let sum_b = (ib as f64 * 2.0) - zb as f64;
                    let de = 2.0 * (ca * sum_a + cb * sum_b);
                    self.table_2c[ia * zbp1 + ib] = (-beta * de).exp().min(1.0);
                }
            }
            self.cached_key_2c = key;
        }
    }

    /// Two-coupling+field branchless table: 2*(z_a+1)*(z_b+1) entries.
    #[inline]
    fn ensure_table_2cf(&mut self, beta: f64, ca: f64, cb: f64, h: f64, za: usize, zb: usize) {
        let key = (beta, ca, cb, h);
        if key != self.cached_key_2cf {
            let zap1 = za + 1;
            let zbp1 = zb + 1;
            let block = zap1 * zbp1;
            for (spin_idx, &spin) in [-1.0_f64, 1.0].iter().enumerate() {
                for ia in 0..zap1 {
                    let sum_a = (ia as f64 * 2.0) - za as f64;
                    for ib in 0..zbp1 {
                        let sum_b = (ib as f64 * 2.0) - zb as f64;
                        let de = 2.0 * spin * (ca * sum_a + cb * sum_b + h);
                        self.table_2cf[spin_idx * block + ia * zbp1 + ib] =
                            (-beta * de).exp().min(1.0);
                    }
                }
            }
            self.cached_key_2cf = key;
        }
    }

    /// Three-coupling spin-symmetry table.
    #[inline]
    fn ensure_table_3c(&mut self, beta: f64, ca: f64, cb: f64, cc: f64) {
        let key = (beta, ca, cb, cc);
        if key != self.cached_key_3c {
            let (zap1, zbp1, zcp1) = (self.z_nn + 1, self.z_nnn + 1, self.z_tnn + 1);
            for ia in 0..zap1 {
                let sa = (ia as f64 * 2.0) - self.z_nn as f64;
                for ib in 0..zbp1 {
                    let sb = (ib as f64 * 2.0) - self.z_nnn as f64;
                    for ic in 0..zcp1 {
                        let sc = (ic as f64 * 2.0) - self.z_tnn as f64;
                        let de = 2.0 * (ca * sa + cb * sb + cc * sc);
                        self.table_3c[ia * zbp1 * zcp1 + ib * zcp1 + ic] =
                            (-beta * de).exp().min(1.0);
                    }
                }
            }
            self.cached_key_3c = key;
        }
    }

    /// Three-coupling+field branchless table.
    #[inline]
    fn ensure_table_3cf(&mut self, beta: f64, ca: f64, cb: f64, cc: f64, h: f64) {
        let key = (beta, ca, cb, cc, h);
        if key != self.cached_key_3cf {
            let (zap1, zbp1, zcp1) = (self.z_nn + 1, self.z_nnn + 1, self.z_tnn + 1);
            let block = zap1 * zbp1 * zcp1;
            for (spin_idx, &spin) in [-1.0_f64, 1.0].iter().enumerate() {
                for ia in 0..zap1 {
                    let sa = (ia as f64 * 2.0) - self.z_nn as f64;
                    for ib in 0..zbp1 {
                        let sb = (ib as f64 * 2.0) - self.z_nnn as f64;
                        for ic in 0..zcp1 {
                            let sc = (ic as f64 * 2.0) - self.z_tnn as f64;
                            let de = 2.0 * spin * (ca * sa + cb * sb + cc * sc + h);
                            self.table_3cf[spin_idx * block + ia * zbp1 * zcp1 + ib * zcp1 + ic] =
                                (-beta * de).exp().min(1.0);
                        }
                    }
                }
            }
            self.cached_key_3cf = key;
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // 15 dedicated sweep methods — direct lattice calls, iter().sum().
    // LLVM monomorphizes per lattice type and unrolls the sum loops.
    // NO function pointers anywhere in the hot path.
    // ══════════════════════════════════════════════════════════════════

    /// J1 only: sign-branch table, skip RNG when dE ≤ 0.
    #[inline]
    fn sweep_j1<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j1: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let z = self.z_nn;
        self.ensure_table_sign(beta, j1, z);
        let table = &self.table_sign;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sum: i32 = lattice.nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let half_de = spin * sum;
            if half_de <= 0 {
                spins[idx] = -spins[idx];
                accepted += 1;
            } else {
                let tidx = (half_de as usize - 1) / 2;
                if rng.gen::<f64>() < table[tidx] {
                    spins[idx] = -spins[idx];
                    accepted += 1;
                }
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J2 only: sign-branch table on NNN.
    #[inline]
    fn sweep_j2<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j2: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let z = self.z_nnn;
        self.ensure_table_sign(beta, j2, z);
        let table = &self.table_sign;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sum: i32 = lattice.next_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let half_de = spin * sum;
            if half_de <= 0 {
                spins[idx] = -spins[idx];
                accepted += 1;
            } else {
                let tidx = (half_de as usize - 1) / 2;
                if rng.gen::<f64>() < table[tidx] {
                    spins[idx] = -spins[idx];
                    accepted += 1;
                }
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J3 only: sign-branch table on TNN.
    #[inline]
    fn sweep_j3<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j3: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let z = self.z_tnn;
        self.ensure_table_sign(beta, j3, z);
        let table = &self.table_sign;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sum: i32 = lattice.third_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let half_de = spin * sum;
            if half_de <= 0 {
                spins[idx] = -spins[idx];
                accepted += 1;
            } else {
                let tidx = (half_de as usize - 1) / 2;
                if rng.gen::<f64>() < table[tidx] {
                    spins[idx] = -spins[idx];
                    accepted += 1;
                }
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// H only: 2-entry table indexed by spin, no neighbor reads.
    #[inline]
    fn sweep_h<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_field(beta, h);
        let table = &self.table_field;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin_idx = ((spins[idx] as i32 + 1) >> 1) as usize;
            if rng.gen::<f64>() < table[spin_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J1 + H: branchless coupling+field table on NN.
    #[inline]
    fn sweep_j1h<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j1: f64, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let z = self.z_nn;
        self.ensure_table_cf(beta, j1, h, z);
        let table = &self.table_cf;
        let zp1 = z + 1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sum: i32 = lattice.nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let spin_idx = ((spin + 1) >> 1) as usize;
            let sum_idx = ((sum + z as i32) / 2) as usize;
            if rng.gen::<f64>() < table[spin_idx * zp1 + sum_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J2 + H: branchless coupling+field table on NNN.
    #[inline]
    fn sweep_j2h<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j2: f64, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let z = self.z_nnn;
        self.ensure_table_cf(beta, j2, h, z);
        let table = &self.table_cf;
        let zp1 = z + 1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sum: i32 = lattice.next_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let spin_idx = ((spin + 1) >> 1) as usize;
            let sum_idx = ((sum + z as i32) / 2) as usize;
            if rng.gen::<f64>() < table[spin_idx * zp1 + sum_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J3 + H: branchless coupling+field table on TNN.
    #[inline]
    fn sweep_j3h<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j3: f64, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let z = self.z_tnn;
        self.ensure_table_cf(beta, j3, h, z);
        let table = &self.table_cf;
        let zp1 = z + 1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sum: i32 = lattice.third_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let spin_idx = ((spin + 1) >> 1) as usize;
            let sum_idx = ((sum + z as i32) / 2) as usize;
            if rng.gen::<f64>() < table[spin_idx * zp1 + sum_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J1 + J2: spin-symmetry two-coupling table on NN+NNN.
    #[inline]
    fn sweep_j1j2<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j1: f64, j2: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let (za, zb) = (self.z_nn, self.z_nnn);
        self.ensure_table_2c(beta, j1, j2, za, zb);
        let table = &self.table_2c;
        let zbp1 = zb + 1;
        let max_idx = (za + 1) * zbp1 - 1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sa: i32 = lattice.nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sb: i32 = lattice.next_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let ai = ((sa + za as i32) / 2) as usize;
            let bi = ((sb + zb as i32) / 2) as usize;
            let base = ai * zbp1 + bi;
            let tidx = if spin > 0 { base } else { max_idx - base };
            if rng.gen::<f64>() < table[tidx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J1 + J3: spin-symmetry two-coupling table on NN+TNN.
    #[inline]
    fn sweep_j1j3<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j1: f64, j3: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let (za, zb) = (self.z_nn, self.z_tnn);
        self.ensure_table_2c(beta, j1, j3, za, zb);
        let table = &self.table_2c;
        let zbp1 = zb + 1;
        let max_idx = (za + 1) * zbp1 - 1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sa: i32 = lattice.nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sb: i32 = lattice.third_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let ai = ((sa + za as i32) / 2) as usize;
            let bi = ((sb + zb as i32) / 2) as usize;
            let base = ai * zbp1 + bi;
            let tidx = if spin > 0 { base } else { max_idx - base };
            if rng.gen::<f64>() < table[tidx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J2 + J3: spin-symmetry two-coupling table on NNN+TNN.
    #[inline]
    fn sweep_j2j3<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j2: f64, j3: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let (za, zb) = (self.z_nnn, self.z_tnn);
        self.ensure_table_2c(beta, j2, j3, za, zb);
        let table = &self.table_2c;
        let zbp1 = zb + 1;
        let max_idx = (za + 1) * zbp1 - 1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sa: i32 = lattice.next_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sb: i32 = lattice.third_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let ai = ((sa + za as i32) / 2) as usize;
            let bi = ((sb + zb as i32) / 2) as usize;
            let base = ai * zbp1 + bi;
            let tidx = if spin > 0 { base } else { max_idx - base };
            if rng.gen::<f64>() < table[tidx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J1 + J2 + H: branchless two-coupling+field table on NN+NNN.
    #[inline]
    fn sweep_j1j2h<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j1: f64, j2: f64, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let (za, zb) = (self.z_nn, self.z_nnn);
        self.ensure_table_2cf(beta, j1, j2, h, za, zb);
        let table = &self.table_2cf;
        let zbp1 = zb + 1;
        let block = (za + 1) * zbp1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sa: i32 = lattice.nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sb: i32 = lattice.next_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let si = ((spin + 1) >> 1) as usize;
            let ai = ((sa + za as i32) / 2) as usize;
            let bi = ((sb + zb as i32) / 2) as usize;
            if rng.gen::<f64>() < table[si * block + ai * zbp1 + bi] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J1 + J3 + H: branchless two-coupling+field table on NN+TNN.
    #[inline]
    fn sweep_j1j3h<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j1: f64, j3: f64, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let (za, zb) = (self.z_nn, self.z_tnn);
        self.ensure_table_2cf(beta, j1, j3, h, za, zb);
        let table = &self.table_2cf;
        let zbp1 = zb + 1;
        let block = (za + 1) * zbp1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sa: i32 = lattice.nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sb: i32 = lattice.third_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let si = ((spin + 1) >> 1) as usize;
            let ai = ((sa + za as i32) / 2) as usize;
            let bi = ((sb + zb as i32) / 2) as usize;
            if rng.gen::<f64>() < table[si * block + ai * zbp1 + bi] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J2 + J3 + H: branchless two-coupling+field table on NNN+TNN.
    #[inline]
    fn sweep_j2j3h<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j2: f64, j3: f64, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        let (za, zb) = (self.z_nnn, self.z_tnn);
        self.ensure_table_2cf(beta, j2, j3, h, za, zb);
        let table = &self.table_2cf;
        let zbp1 = zb + 1;
        let block = (za + 1) * zbp1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sa: i32 = lattice.next_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sb: i32 = lattice.third_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let si = ((spin + 1) >> 1) as usize;
            let ai = ((sa + za as i32) / 2) as usize;
            let bi = ((sb + zb as i32) / 2) as usize;
            if rng.gen::<f64>() < table[si * block + ai * zbp1 + bi] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J1 + J2 + J3: spin-symmetry three-coupling table.
    #[inline]
    fn sweep_j1j2j3<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j1: f64, j2: f64, j3: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_3c(beta, j1, j2, j3);
        let table = &self.table_3c;
        let (zap1, zbp1, zcp1) = (self.z_nn + 1, self.z_nnn + 1, self.z_tnn + 1);
        let max_idx = zap1 * zbp1 * zcp1 - 1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sa: i32 = lattice.nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sb: i32 = lattice.next_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sc: i32 = lattice.third_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let ai = ((sa + self.z_nn as i32) / 2) as usize;
            let bi = ((sb + self.z_nnn as i32) / 2) as usize;
            let ci = ((sc + self.z_tnn as i32) / 2) as usize;
            let base = ai * zbp1 * zcp1 + bi * zcp1 + ci;
            let tidx = if spin > 0 { base } else { max_idx - base };
            if rng.gen::<f64>() < table[tidx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }

    /// J1 + J2 + J3 + H: branchless three-coupling+field table.
    #[inline]
    fn sweep_j1j2j3h<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L, j1: f64, j2: f64, j3: f64, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_3cf(beta, j1, j2, j3, h);
        let table = &self.table_3cf;
        let (zap1, zbp1, zcp1) = (self.z_nn + 1, self.z_nnn + 1, self.z_tnn + 1);
        let block = zap1 * zbp1 * zcp1;
        let n = lattice.num_sites();
        let mut accepted = 0;
        for idx in 0..n {
            let spin = spins[idx] as i32;
            let sa: i32 = lattice.nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sb: i32 = lattice.next_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let sc: i32 = lattice.third_nearest_neighbors(idx).iter().map(|&i| spins[i] as i32).sum();
            let si = ((spin + 1) >> 1) as usize;
            let ai = ((sa + self.z_nn as i32) / 2) as usize;
            let bi = ((sb + self.z_nnn as i32) / 2) as usize;
            let ci = ((sc + self.z_tnn as i32) / 2) as usize;
            if rng.gen::<f64>() < table[si * block + ai * zbp1 * zcp1 + bi * zcp1 + ci] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }
        SweepResult { accepted, attempted: n }
    }
}

impl McAlgorithm for Metropolis {
    fn sweep<L: Lattice, R: Rng>(
        &mut self, spins: &mut [i8], lattice: &L,
        j1: f64, j2: f64, j3: f64, h: f64, beta: f64, rng: &mut R,
    ) -> SweepResult {
        match self.strategy {
            SweepStrategy::J1 => self.sweep_j1(spins, lattice, j1, beta, rng),
            SweepStrategy::J2 => self.sweep_j2(spins, lattice, j2, beta, rng),
            SweepStrategy::J3 => self.sweep_j3(spins, lattice, j3, beta, rng),
            SweepStrategy::H => self.sweep_h(spins, lattice, h, beta, rng),
            SweepStrategy::J1H => self.sweep_j1h(spins, lattice, j1, h, beta, rng),
            SweepStrategy::J2H => self.sweep_j2h(spins, lattice, j2, h, beta, rng),
            SweepStrategy::J3H => self.sweep_j3h(spins, lattice, j3, h, beta, rng),
            SweepStrategy::J1J2 => self.sweep_j1j2(spins, lattice, j1, j2, beta, rng),
            SweepStrategy::J1J3 => self.sweep_j1j3(spins, lattice, j1, j3, beta, rng),
            SweepStrategy::J2J3 => self.sweep_j2j3(spins, lattice, j2, j3, beta, rng),
            SweepStrategy::J1J2H => self.sweep_j1j2h(spins, lattice, j1, j2, h, beta, rng),
            SweepStrategy::J1J3H => self.sweep_j1j3h(spins, lattice, j1, j3, h, beta, rng),
            SweepStrategy::J2J3H => self.sweep_j2j3h(spins, lattice, j2, j3, h, beta, rng),
            SweepStrategy::J1J2J3 => self.sweep_j1j2j3(spins, lattice, j1, j2, j3, beta, rng),
            SweepStrategy::J1J2J3H => self.sweep_j1j2j3h(spins, lattice, j1, j2, j3, h, beta, rng),
        }
    }

    fn name(&self) -> &'static str {
        "Metropolis"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::square::SquareLattice;
    use crate::rng::create_rng;

    fn all_up_spins(n: usize) -> Vec<i8> {
        vec![1; n]
    }

    fn random_spins(lattice: &SquareLattice, rng: &mut impl Rng) -> Vec<i8> {
        (0..lattice.num_sites())
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect()
    }

    fn compute_energy(spins: &[i8], lattice: &SquareLattice, j1: f64, j2: f64, j3: f64, h: f64) -> f64 {
        crate::observables::energy_per_site(spins, lattice, j1, j2, j3, h)
    }

    fn make_metro(j1: f64, j2: f64, j3: f64, h: f64) -> Metropolis {
        // Square lattice: z_nn=4, z_nnn=4, z_tnn=4
        Metropolis::new(j1, j2, j3, h, 4, 4, 4)
    }

    // ── Basic tests ───────────────────────────────────────────────────

    #[test]
    fn test_metropolis_name() {
        assert_eq!(make_metro(1.0, 0.0, 0.0, 0.0).name(), "Metropolis");
    }

    #[test]
    fn test_sweep_returns_valid_result() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let result = make_metro(1.0, 0.0, 0.0, 0.0).sweep(
            &mut spins, &lattice, 1.0, 0.0, 0.0, 0.0, 1.0, &mut rng,
        );
        assert_eq!(result.attempted, 16);
        assert!(result.accepted <= result.attempted);
    }

    // ── Determinism tests (one per strategy) ──────────────────────────

    fn assert_deterministic(j1: f64, j2: f64, j3: f64, h: f64) {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins1 = all_up_spins(lattice.num_sites());
        let mut spins2 = all_up_spins(lattice.num_sites());
        let mut rng1 = create_rng(123);
        let mut rng2 = create_rng(123);
        make_metro(j1, j2, j3, h).sweep(&mut spins1, &lattice, j1, j2, j3, h, 0.5, &mut rng1);
        make_metro(j1, j2, j3, h).sweep(&mut spins2, &lattice, j1, j2, j3, h, 0.5, &mut rng2);
        assert_eq!(spins1, spins2);
    }

    #[test] fn test_deterministic_j1()      { assert_deterministic(1.0, 0.0, 0.0, 0.0); }
    #[test] fn test_deterministic_j2()      { assert_deterministic(0.0, 0.5, 0.0, 0.0); }
    #[test] fn test_deterministic_j3()      { assert_deterministic(0.0, 0.0, 0.5, 0.0); }
    #[test] fn test_deterministic_h()       { assert_deterministic(0.0, 0.0, 0.0, 0.5); }
    #[test] fn test_deterministic_j1h()     { assert_deterministic(1.0, 0.0, 0.0, 0.5); }
    #[test] fn test_deterministic_j2h()     { assert_deterministic(0.0, 0.5, 0.0, 0.3); }
    #[test] fn test_deterministic_j3h()     { assert_deterministic(0.0, 0.0, 0.5, 0.3); }
    #[test] fn test_deterministic_j1j2()    { assert_deterministic(1.0, 0.3, 0.0, 0.0); }
    #[test] fn test_deterministic_j1j3()    { assert_deterministic(1.0, 0.0, 0.3, 0.0); }
    #[test] fn test_deterministic_j2j3()    { assert_deterministic(0.0, 0.5, 0.3, 0.0); }
    #[test] fn test_deterministic_j1j2h()   { assert_deterministic(1.0, 0.3, 0.0, 0.5); }
    #[test] fn test_deterministic_j1j3h()   { assert_deterministic(1.0, 0.0, 0.3, 0.5); }
    #[test] fn test_deterministic_j2j3h()   { assert_deterministic(0.0, 0.5, 0.3, 0.5); }
    #[test] fn test_deterministic_j1j2j3()  { assert_deterministic(1.0, 0.3, 0.2, 0.0); }
    #[test] fn test_deterministic_j1j2j3h() { assert_deterministic(1.0, 0.3, 0.2, 0.5); }

    // ── Physics tests ─────────────────────────────────────────────────

    fn assert_energy_decreases_at_zero_t(j1: f64, j2: f64, j3: f64, h: f64) {
        let lattice = SquareLattice::new(8).unwrap();
        let mut rng = create_rng(42);
        let mut spins = random_spins(&lattice, &mut rng);
        let energy_before = compute_energy(&spins, &lattice, j1, j2, j3, h);
        for _ in 0..100 {
            make_metro(j1, j2, j3, h).sweep(&mut spins, &lattice, j1, j2, j3, h, 1e10, &mut rng);
        }
        let energy_after = compute_energy(&spins, &lattice, j1, j2, j3, h);
        assert!(energy_after <= energy_before + 1e-10,
            "Energy should not increase at T=0: before={energy_before}, after={energy_after}");
    }

    #[test] fn test_energy_decreases_j1()      { assert_energy_decreases_at_zero_t(1.0, 0.0, 0.0, 0.0); }
    #[test] fn test_energy_decreases_j2()      { assert_energy_decreases_at_zero_t(0.0, 1.0, 0.0, 0.0); }
    #[test] fn test_energy_decreases_j3()      { assert_energy_decreases_at_zero_t(0.0, 0.0, 1.0, 0.0); }
    #[test] fn test_energy_decreases_j1h()     { assert_energy_decreases_at_zero_t(1.0, 0.0, 0.0, 0.5); }
    #[test] fn test_energy_decreases_j2h()     { assert_energy_decreases_at_zero_t(0.0, 0.5, 0.0, 0.3); }
    #[test] fn test_energy_decreases_j1j2()    { assert_energy_decreases_at_zero_t(1.0, 0.5, 0.0, 0.0); }
    #[test] fn test_energy_decreases_j1j3()    { assert_energy_decreases_at_zero_t(1.0, 0.0, 0.5, 0.0); }
    #[test] fn test_energy_decreases_j1j2j3()  { assert_energy_decreases_at_zero_t(1.0, 0.3, 0.2, 0.0); }
    #[test] fn test_energy_decreases_j1j2j3h() { assert_energy_decreases_at_zero_t(1.0, 0.3, 0.2, 0.5); }

    #[test]
    fn test_high_temperature_high_acceptance() {
        let lattice = SquareLattice::new(8).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let result = make_metro(1.0, 0.0, 0.0, 0.0).sweep(
            &mut spins, &lattice, 1.0, 0.0, 0.0, 0.0, 0.001, &mut rng,
        );
        assert!(result.acceptance_rate() > 0.5);
    }

    #[test]
    fn test_all_up_ground_state_stable_at_low_t() {
        let lattice = SquareLattice::new(8).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        for _ in 0..10 {
            make_metro(1.0, 0.0, 0.0, 0.0).sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, 0.0, 10.0, &mut rng);
        }
        let mag: f64 = spins.iter().map(|&s| f64::from(s)).sum::<f64>() / spins.len() as f64;
        assert!(mag > 0.9, "Ground state should remain magnetized, got m={mag}");
    }

    #[test]
    fn test_h_only_aligns_spins() {
        let lattice = SquareLattice::new(8).unwrap();
        let mut rng = create_rng(42);
        let mut spins = random_spins(&lattice, &mut rng);
        for _ in 0..200 {
            make_metro(0.0, 0.0, 0.0, 1.0).sweep(&mut spins, &lattice, 0.0, 0.0, 0.0, 1.0, 100.0, &mut rng);
        }
        let mag: f64 = spins.iter().map(|&s| f64::from(s)).sum::<f64>() / spins.len() as f64;
        assert!(mag > 0.99, "Field-only at low T should align spins: got m={mag}");
    }

    #[test]
    fn test_energy_all_up_with_j3() {
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let e = compute_energy(&spins, &lattice, 1.0, 0.0, 0.5, 0.0);
        assert!((e - (-3.0)).abs() < 1e-10, "Expected -3.0, got {e}");
    }

    #[test]
    fn test_energy_all_up_j1j2j3() {
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let e = compute_energy(&spins, &lattice, 1.0, 0.5, 0.25, 0.0);
        assert!((e - (-3.5)).abs() < 1e-10, "Expected -3.5, got {e}");
    }

    // ── Cross-lattice tests (non-square coordination numbers) ────────

    fn assert_energy_decreases_on_lattice<L: Lattice>(
        lattice: &L, j1: f64, j2: f64, j3: f64, h: f64,
        z_nn: usize, z_nnn: usize, z_tnn: usize,
    ) {
        let mut rng = create_rng(42);
        let mut spins: Vec<i8> = (0..lattice.num_sites())
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        let e_before = crate::observables::energy_per_site(&spins, lattice, j1, j2, j3, h);
        let mut metro = Metropolis::new(j1, j2, j3, h, z_nn, z_nnn, z_tnn);
        for _ in 0..100 {
            metro.sweep(&mut spins, lattice, j1, j2, j3, h, 1e10, &mut rng);
        }
        let e_after = crate::observables::energy_per_site(&spins, lattice, j1, j2, j3, h);
        assert!(e_after <= e_before + 1e-10,
            "Energy should decrease: {e_before} -> {e_after}");
    }

    #[test]
    fn test_energy_decreases_j1_triangular() {
        use crate::lattice::triangular::TriangularLattice;
        let lat = TriangularLattice::new(8).unwrap();
        assert_energy_decreases_on_lattice(&lat, 1.0, 0.0, 0.0, 0.0, 6, 6, 6);
    }

    #[test]
    fn test_energy_decreases_j1_chain() {
        use crate::lattice::chain::ChainLattice;
        let lat = ChainLattice::new(50).unwrap();
        assert_energy_decreases_on_lattice(&lat, 1.0, 0.0, 0.0, 0.0, 2, 2, 2);
    }

    #[test]
    fn test_energy_decreases_j1_honeycomb() {
        use crate::lattice::honeycomb::HoneycombLattice;
        let lat = HoneycombLattice::new(8).unwrap();
        assert_energy_decreases_on_lattice(&lat, 1.0, 0.0, 0.0, 0.0, 3, 6, 3);
    }

    #[test]
    fn test_energy_decreases_j1j2_cubic() {
        use crate::lattice::cubic::CubicLattice;
        let lat = CubicLattice::new(6).unwrap();
        assert_energy_decreases_on_lattice(&lat, 1.0, 0.5, 0.0, 0.0, 6, 12, 8);
    }
}
