// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use crate::integer_arith::ArithUtils;
use crate::utils::reverse_bits_perm;
use std::sync::Arc;

/// Holds the context information for RqPolys, including degree n, modulus q, and optionally precomputed
/// roots of unity for NTT purposes.
#[derive(Debug)]
pub(crate) struct RqPolyContext<T> {
    pub n: usize,
    pub q: T,
    pub is_ntt_enabled: bool,
    pub roots: Vec<T>,
    pub invroots: Vec<T>,
}

/// Polynomials in Rq = Zq[x]/(x^n + 1).
#[derive(Clone, Debug)]
pub struct RqPoly<T> {
    context: Arc<RqPolyContext<T>>,
    pub coeffs: Vec<T>,
    pub is_ntt_form: bool,
}

/// Number-theoretic transform (NTT) and fast polynomial multiplication based on NTT.
pub trait NTT<T>: Clone {
    fn is_ntt_form(&self) -> bool;

    fn set_ntt_form(&mut self, value: bool);

    fn forward_transform(&mut self);

    fn inverse_transform(&mut self);

    fn coeffwise_multiply(&self, other: &Self) -> Self;

    fn multiply_fast(&self, other: &Self) -> Self;
}

/// Arithmetics on general ring elements.
pub trait FiniteRingElt {
    fn add_inplace(&mut self, other: &Self);

    fn sub_inplace(&mut self, other: &Self);

    fn negate_inplace(&mut self);

    fn multiply(&self, other: &Self) -> Self;
}

impl<T> RqPolyContext<T>
where
    T: ArithUtils<T> + PartialEq + Clone,
{
    pub fn new(n: usize, q: &T) -> Self {
        let mut a = RqPolyContext {
            n: n,
            q: q.clone(),
            is_ntt_enabled: false,
            invroots: vec![],
            roots: vec![],
        };
        a.compute_roots();
        a
    }

    fn compute_roots(&mut self) {
        let mut roots = vec![];

        let root = self.find_root();
        if root.is_none() {
            self.is_ntt_enabled = false;
            return;
        }
        self.is_ntt_enabled = true;
        let phi = root.unwrap();

        let mut s = T::one();
        for _ in 0..self.n {
            roots.push(s.clone());
            s = T::mul_mod(&s, &phi, &self.q);
        }
        // now bit reverse a vector

        reverse_bits_perm(&mut roots);
        self.roots = roots;

        let mut invroots: Vec<T> = vec![];
        for x in self.roots.iter() {
            invroots.push(T::inv_mod(x, &self.q));
        }
        self.invroots = invroots;
    }

    pub fn find_root(&self) -> Option<T> {
        let bign = T::from_u32_raw(self.n as u32);
        let q_minus_one = T::sub(&self.q, &T::one());

        let power = T::div(&q_minus_one, &T::double(&bign));
        let mut s = T::one();
        let one = T::one();
        s = T::modulus(&s, &self.q);
        let mut spow = T::pow_mod(&s, &power, &self.q);
        let max_iter = 100;
        let mut iter = 0;
        while T::pow_mod(&spow, &bign, &self.q) != q_minus_one && iter < max_iter {
            s = T::add_mod(&s, &one, &self.q);
            // s = T::sample_blw(&self.q);
            spow = T::pow_mod(&s, &power, &self.q);
            iter += 1;
        }
        if iter < max_iter {
            Some(spow)
        } else {
            None
        }
    }
}

// NTT implementation
impl<T> NTT<T> for RqPoly<T>
where
    T: ArithUtils<T> + Clone,
{
    fn is_ntt_form(&self) -> bool {
        self.is_ntt_form
    }

    fn set_ntt_form(&mut self, value: bool) {
        self.is_ntt_form = value;
    }

    fn forward_transform(&mut self) {
        if self.is_ntt_form {
            panic!("is already in ntt");
        }

        let n = self.context.n;
        let q = self.context.q.clone();

        let mut t = n;
        let mut m = 1;
        while m < n {
            t >>= 1;
            for i in 0..m {
                let j1 = 2 * i * t;
                let j2 = j1 + t - 1;
                let phi = &self.context.roots[m + i];
                for j in j1..j2 + 1 {
                    let x = T::mul_mod(&self.coeffs[j + t], &phi, &q);
                    self.coeffs[j + t] = T::sub_mod(&self.coeffs[j], &x, &q);
                    self.coeffs[j] = T::add_mod(&self.coeffs[j], &x, &q);
                }
            }
            m <<= 1;
        }
        self.set_ntt_form(true);
    }

    fn inverse_transform(&mut self) {
        if !self.is_ntt_form {
            panic!("is already not in ntt");
        }
        let n = self.context.n;
        let q = self.context.q.clone();

        let mut t = 1;
        let mut m = n;
        let ninv = T::inv_mod(&T::from_u32(n as u32, &q), &q);
        while m > 1 {
            let mut j1 = 0;
            let h = m >> 1;
            for i in 0..h {
                let j2 = j1 + t - 1;
                let s = &self.context.invroots[h + i];
                for j in j1..j2 + 1 {
                    let u = self.coeffs[j].clone();
                    let v = self.coeffs[j + t].clone();
                    self.coeffs[j] = T::add_mod(&u, &v, &q);

                    let tmp = T::sub_mod(&u, &v, &q);
                    self.coeffs[j + t] = T::mul_mod(&tmp, &s, &q);
                }
                j1 += 2 * t;
            }
            t <<= 1;
            m >>= 1;
        }
        for x in 0..n {
            self.coeffs[x] = T::mul_mod(&ninv, &self.coeffs[x], &q);
        }
        self.set_ntt_form(false);
    }

    fn coeffwise_multiply(&self, other: &Self) -> Self {
        let mut c = self.clone();
        for (inputs, cc) in self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .zip(c.coeffs.iter_mut())
        {
            *cc = T::mul_mod(inputs.0, inputs.1, &self.context.q);
        }
        c
    }

    fn multiply_fast(&self, other: &Self) -> Self {
        let mut a: Self = self.clone();
        let mut b = other.clone();

        if !a.is_ntt_form {
            a.forward_transform();
        }
        if !b.is_ntt_form {
            b.forward_transform();
        }
        let mut c = a.coeffwise_multiply(&b);
        c.inverse_transform();
        c
    }
}

impl<T> FiniteRingElt for RqPoly<T>
where
    T: ArithUtils<T> + Clone,
{
    fn add_inplace(&mut self, other: &Self) {
        let iter = self.coeffs.iter_mut().zip(other.coeffs.iter());
        for (x, y) in iter {
            *x = T::add_mod(x, y, &self.context.q);
        }
    }

    fn sub_inplace(&mut self, other: &Self) {
        let iter = self.coeffs.iter_mut().zip(other.coeffs.iter());
        for (x, y) in iter {
            *x = T::sub_mod(x, y, &self.context.q);
        }
    }

    fn negate_inplace(&mut self) {
        for x in self.coeffs.iter_mut() {
            *x = T::sub_mod(&self.context.q, x, &self.context.q);
        }
    }

    // naive multiplication
    fn multiply(&self, other: &Self) -> Self {
        let f = &self.coeffs;
        let g = &other.coeffs;
        let n = self.context.n;
        let q = self.context.q.clone();
        let mut res = vec![T::zero(); n];

        for i in 0..n {
            for j in 0..i + 1 {
                let tmp = T::mul_mod(&f[j], &g[i - j], &q);
                res[i] = T::add_mod(&res[i], &tmp, &q);
            }
            for j in i + 1..self.context.n {
                let tmp = T::mul_mod(&f[j], &g[n + i - j], &q);
                res[i] = T::sub_mod(&res[i], &tmp, &q);
            }
            res[i] = T::modulus(&res[i], &q);
        }
        RqPoly {
            coeffs: res,
            is_ntt_form: false,
            context: self.context.clone(),
        }
    }
}

/// Utility functions for generating random polynomials.
pub(crate) mod randutils {
    use rand::distributions::{Distribution, Normal};
    use rand::rngs::{OsRng, StdRng};
    use rand::FromEntropy;
    use rand::{thread_rng, Rng};
    use super::*;

    pub(crate) fn sample_ternary_poly<T>(context: Arc<RqPolyContext<T>>) -> RqPoly<T>
    where
        T: ArithUtils<T>,
    {
        let mut rng = OsRng::new().unwrap();
        let mut c = vec![];
        for _x in 0..context.n {
            let t = rng.gen_range(-1i32, 2i32);
            if t >= 0 {
                c.push(T::from_u32_raw(t as u32));
            } else {
                c.push(T::sub(&context.q, &T::one()));
            }
        }
        RqPoly {
            coeffs: c,
            is_ntt_form: false,
            context: context.clone(),
        }
    }

    pub(crate) fn sample_ternary_poly_prng<T>(context: Arc<RqPolyContext<T>>) -> RqPoly<T>
    where
        T: ArithUtils<T>,
    {
        let mut rng = StdRng::from_entropy();
        let mut c = vec![];
        for _x in 0..context.n {
            let t = rng.gen_range(-1i32, 2i32);
            if t >= 0 {
                c.push(T::from_u32_raw(t as u32));
            } else {
                c.push(T::sub(&context.q, &T::one()));
            }
        }
        RqPoly {
            coeffs: c,
            is_ntt_form: false,
            context: context,
        }
    }

    /// Sample a polynomial with Gaussian coefficients in the ring Rq.
    pub(crate) fn sample_gaussian_poly<T>(context: Arc<RqPolyContext<T>>, stdev: f64) -> RqPoly<T>
    where
        T: ArithUtils<T>,
    {
        let mut c = vec![];
        let normal = Normal::new(0.0, stdev);
        let mut rng = thread_rng();
        for _ in 0..context.n {
            let tmp = normal.sample(&mut rng);

            // branch on sign
            if tmp >= 0.0 {
                c.push(T::from_u64_raw(tmp as u64));
            } else {
                let neg = T::from_u64_raw(-tmp as u64);
                c.push(T::sub(&context.q, &neg));
            }
        }
        RqPoly {
            coeffs: c,
            is_ntt_form: false,
            context: context,
        }
    }

    /// Sample a uniform polynomial in the ring Rq.
    pub(crate) fn sample_uniform_poly<T>(context: Arc<RqPolyContext<T>>) -> RqPoly<T>
    where
        T: ArithUtils<T>,
    {
        let mut c = vec![];
        let mut rng = StdRng::from_entropy();
        for _x in 0..context.n {
            c.push(T::sample_below_from_rng(&context.q, &mut rng));
        }
        RqPoly {
            coeffs: c,
            is_ntt_form: false,
            context: context.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer_arith::scalar::Scalar;

    fn from_vec<T>(v: &Vec<u32>, context: Arc<RqPolyContext<T>>) -> RqPoly<T>
    where
        T: ArithUtils<T>,
    {
        let mut c = vec![];
        for _x in 0..context.n {
            let tmp = T::from_u32_raw(v[_x]);
            c.push(T::modulus(&tmp, &context.q));
        }
        RqPoly {
            coeffs: c,
            is_ntt_form: false,
            context: context.clone(),
        }
    }
    #[test]
    fn test_ntt_constant_scalar() {
        let q = Scalar::new_modulus(18014398492704769u64);
        let context = Arc::new(RqPolyContext::new(8, &q));
        let v: Vec<u32> = vec![0; 8];

        let mut testpoly = from_vec(&v, context.clone());

        let tmp = Scalar::modulus(&Scalar::from_u32_raw(101), &q);
        testpoly.coeffs[0] = tmp.clone();

        testpoly.forward_transform();

        for i in 0..context.n {
            assert_eq!(testpoly.coeffs[i], tmp);
        }
    }

    #[test]
    fn test_ntt_scalar_compose_inverse() {
        let q = Scalar::new_modulus(18014398492704769u64);
        let context = RqPolyContext::new(2048, &q);
        let arc = Arc::new(context);
        let a = randutils::sample_uniform_poly(arc.clone());
        let mut aa = a.clone();
        aa.forward_transform();
        aa.inverse_transform();

        assert_eq!(a.coeffs, aa.coeffs);

        // first inverse, then forward
        aa.set_ntt_form(true);

        aa.inverse_transform();

        aa.forward_transform();

        assert_eq!(a.coeffs, aa.coeffs);
    }

    #[test]
    fn test_fast_multiply_with_one() {
        let context = RqPolyContext::new(4, &Scalar::new_modulus(12289));
        let arc = Arc::new(context);
        let a = from_vec(&vec![1, 0, 0, 0], arc.clone());
        let b = from_vec(&vec![3, 4, 5, 6], arc.clone());
        let c1 = a.multiply_fast(&b);
        assert_eq!(b.coeffs, c1.coeffs);
    }

    #[test]
    fn test_fast_multiply_scalar_2048() {
        let q = Scalar::new_modulus(18014398492704769u64);
        let context = RqPolyContext::new(2048, &q);
        let arc = Arc::new(context);

        let a = randutils::sample_uniform_poly(arc.clone());
        let b = randutils::sample_uniform_poly(arc.clone());
        let c = a.multiply(&b);
        let c1 = a.multiply_fast(&b);
        assert_eq!(c.coeffs, c1.coeffs);
    }

    #[test]
    fn test_find_root_scalar(){
        let context2 = RqPolyContext::new(4, &Scalar::new_modulus(12289));
        assert_eq!(context2.find_root().unwrap(), Scalar::from_u64_raw(8246u64));
    }
}
