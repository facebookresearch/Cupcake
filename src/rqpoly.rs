// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use crate::integer_arith::{SuperTrait, ArithUtils};
use crate::integer_arith::butterfly::{inverse_butterfly,butterfly};
use crate::polyarith::lazy_ntt::{lazy_ntt_u64, lazy_inverse_ntt_u64};
use crate::integer_arith::util::compute_harvey_ratio; 
use crate::utils::reverse_bits_perm;
use std::sync::Arc;
use crate::traits::*; 

/// Holds the context information for RqPolys, including degree n, modulus q, and optionally precomputed
/// roots of unity for NTT purposes.
#[derive(Debug)]
pub struct RqPolyContext<T> {
    pub n: usize,
    pub q: T,
    pub is_ntt_enabled: bool,
    pub roots: Vec<T>,
    pub invroots: Vec<T>,
    pub scaled_roots: Vec<T>, // for use in lazy ntt
    pub scaled_invroots: Vec<T>, // for use in lazy inverse ntt
}

/// Polynomials in Rq = Zq[x]/(x^n + 1).
#[derive(Clone, Debug)]
pub struct RqPoly<T> {
    pub(crate) context: Option<Arc<RqPolyContext<T>>>,
    pub coeffs: Vec<T>,
    pub is_ntt_form: bool,
}

impl<T> RqPoly<T> where T:Clone{
    pub fn new_without_context(coeffs: &[T], is_ntt_form:bool) -> Self{
        RqPoly{
            context: None,
            coeffs: coeffs.to_vec(),
            is_ntt_form,
        }
    }
}

impl<T> RqPoly<T> where T:Clone + ArithUtils<T>{
    pub fn new(context: Arc<RqPolyContext<T>>) -> Self{
        let n = context.n; 
        RqPoly{
            context: Some(context), 
            coeffs: vec![T::zero(); n], 
            is_ntt_form: false
        }
    }

    pub(crate) fn set_context(&mut self, context: Arc<RqPolyContext<T>>){
        self.context = Some(context);
    }
}

impl<T> PartialEq for RqPoly<T> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        if self.coeffs.len() != other.coeffs.len() {
            return false
        }
        for i in 0..self.coeffs.len(){
            if self.coeffs[i] != other.coeffs[i] { return false; }
        }
        if self.is_ntt_form != other.is_ntt_form { return false; }
         true
    }
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
    T: SuperTrait<T>,
{
    pub fn new(n: usize, q: &T) -> Self {
        let mut a = RqPolyContext {
            n,
            q: q.clone(),
            is_ntt_enabled: false,
            invroots: vec![],
            roots: vec![],
            scaled_roots: vec![], 
            scaled_invroots: vec![], 
        };
        a.compute_roots();
        a.compute_scaled_roots();

        a
    }

    fn compute_scaled_roots(&mut self){
        if !self.is_ntt_enabled{
            return; 
        }
        // compute scaled roots as wiprime = wi
        for i in 0..self.n {
            self.scaled_roots.push(T::from(compute_harvey_ratio(self.roots[i].rep(), self.q.rep()))); 
        }

        for i in 0..self.n {
            self.scaled_invroots.push(T::from(compute_harvey_ratio(self.invroots[i].rep(), self.q.rep()))); 
        }
    }

    fn compute_roots(&mut self) {
        let mut roots = vec![];

        let root = self.find_root();
        if root.is_none() {
            self.is_ntt_enabled = false;
            return;
        }
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
        self.is_ntt_enabled = true;
    }

    pub fn find_root(&self) -> Option<T> {
        let bign = T::from(self.n as u32);
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

impl<T> RqPoly<T>
where
    T: SuperTrait<T>
{
    fn lazy_ntt(&mut self)
    {
        let context = self.context.as_ref().unwrap();
        if self.is_ntt_form {
            panic!("is already in ntt");
        }
        let q = context.q.rep();

        let mut coeffs_u64: Vec<u64> = self.coeffs.iter()
        .map(|elm| elm.rep())
        .collect();

        let roots_u64: Vec<u64> = context.roots.iter()
        .map(|elm| elm.rep())
        .collect();
        let scaledroots_u64: Vec<u64> = context.scaled_roots.iter()
        .map(|elm| elm.rep())
        .collect();

        lazy_ntt_u64(&mut coeffs_u64, &roots_u64, &scaledroots_u64, q); 

        for (coeff, coeff_u64) in self.coeffs.iter_mut().zip(coeffs_u64.iter()){
            *coeff = T::modulus(&T::from(*coeff_u64), &context.q); 
        }
        self.set_ntt_form(true);
    }

    fn lazy_inverse_ntt(&mut self){
        let context = self.context.as_ref().unwrap();
        if !self.is_ntt_form {
            panic!("is already not in ntt");
        }
        let n = context.n;
        let q = context.q.clone();
        let ninv = T::inv_mod(&T::from_u32(n as u32, &q), &q);

        let mut coeffs_u64: Vec<u64> = self.coeffs.iter()
        .map(|elm| elm.rep())
        .collect();

        let invroots_u64: Vec<u64> = context.invroots.iter()
        .map(|elm| elm.rep())
        .collect();

        let scaled_invroots_u64: Vec<u64> = context.scaled_invroots.iter()
        .map(|elm| elm.rep())
        .collect();

        lazy_inverse_ntt_u64(&mut coeffs_u64, &invroots_u64, &scaled_invroots_u64, q.rep()); 

        for (coeff, coeff_u64) in self.coeffs.iter_mut().zip(coeffs_u64.iter()){
            *coeff = T::mul_mod(&ninv, &T::from(*coeff_u64), &context.q); 
        }
        self.set_ntt_form(false);
    }
}

// NTT implementation(lazy version)
#[cfg(feature = "lazy_ntt")]
impl<T> NTT<T> for RqPoly<T>
where
    T: SuperTrait<T>
{
    fn is_ntt_form(&self) -> bool {
        self.is_ntt_form
    }

    fn set_ntt_form(&mut self, value: bool) {
        self.is_ntt_form = value;
    }

    fn forward_transform(&mut self) {
        self.lazy_ntt()
    }

    fn inverse_transform(&mut self) {
        self.lazy_inverse_ntt()
    }
}

// NTT implementation
#[cfg(not(feature = "lazy_ntt"))]
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
        let context = self.context.as_ref().unwrap();
        if self.is_ntt_form {
            panic!("is already in ntt");
        }

        let n = context.n;
        let q = context.q.clone();

        let mut t = n;
        let mut m = 1;
        while m < n {
            t >>= 1;
            for i in 0..m {
                let j1 = 2 * i * t;
                let j2 = j1 + t - 1;
                let phi = &context.roots[m + i];
                for j in j1..j2 + 1 {
                    // butteffly: 
                    let (a, b) = self.coeffs.split_at_mut(j+1);
                    butterfly(&mut a[j], &mut b[t-1], phi, &q);
                }
            }
            m <<= 1;
        }
        self.set_ntt_form(true);
    }

    fn inverse_transform(&mut self) {
        let context = self.context.as_ref().unwrap();
        if !self.is_ntt_form {
            panic!("is already not in ntt");
        }
        let n = context.n;
        let q = context.q.clone();

        let mut t = 1;
        let mut m = n;
        let ninv = T::inv_mod(&T::from_u32(n as u32, &q), &q);
        while m > 1 {
            let mut j1 = 0;
            let h = m >> 1;
            for i in 0..h {
                let j2 = j1 + t - 1;
                let s = &context.invroots[h + i];
                for j in j1..j2 + 1 {
                    // inverse butterfly
                    let (a, b) = self.coeffs.split_at_mut(j+1);
                    inverse_butterfly(&mut a[j], &mut b[t-1], s, &q);
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
}

impl<T> FastPolyMultiply<T> for RqPoly<T>
where T: SuperTrait<T>{
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

    fn coeffwise_multiply(&self, other: &Self) -> Self {
        let context = self.context.as_ref().unwrap();
        let mut c = self.clone();
        for (inputs, cc) in self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .zip(c.coeffs.iter_mut())
        {
            *cc = T::mul_mod(inputs.0, inputs.1, &context.q);
        }
        c
    }
}


impl<T> FiniteRingElt for RqPoly<T>
where
    T: ArithUtils<T> + Clone,
{
    fn add_inplace(&mut self, other: &Self) {
        let context = self.context.as_ref().unwrap();
        let iter = self.coeffs.iter_mut().zip(other.coeffs.iter());
        for (x, y) in iter {
            *x = T::add_mod(x, y, &context.q);
        }
    }

    fn sub_inplace(&mut self, other: &Self) {
        let context = self.context.as_ref().unwrap();
        let iter = self.coeffs.iter_mut().zip(other.coeffs.iter());
        for (x, y) in iter {
            *x = T::sub_mod(x, y, &context.q);
        }
    }

    fn negate_inplace(&mut self) {
        let context = self.context.as_ref().unwrap();
        for x in self.coeffs.iter_mut() {
            *x = T::sub_mod(&context.q, x, &context.q);
        }
    }

    // naive multiplication
    fn multiply(&self, other: &Self) -> Self {
        // check context exists
        let context = self.context.as_ref().unwrap();
        let f = &self.coeffs;
        let g = &other.coeffs;
        let n = context.n;
        let q = context.q.clone();
        let mut res = vec![T::zero(); n];

        for i in 0..n {
            for j in 0..i + 1 {
                let tmp = T::mul_mod(&f[j], &g[i - j], &q);
                res[i] = T::add_mod(&res[i], &tmp, &q);
            }
            for j in i + 1..context.n {
                let tmp = T::mul_mod(&f[j], &g[n + i - j], &q);
                res[i] = T::sub_mod(&res[i], &tmp, &q);
            }
            res[i] = T::modulus(&res[i], &q);
        }
        RqPoly {
            coeffs: res,
            is_ntt_form: false,
            context: Some(context.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer_arith::scalar::Scalar;

    fn from_vec<T>(v: &Vec<u32>, context: Arc<RqPolyContext<T>>) -> RqPoly<T>
    where
        T: ArithUtils<T> + From<u32>,
    {
        let mut c = vec![];
        for _x in 0..context.n {
            let tmp = T::from(v[_x]);
            c.push(T::modulus(&tmp, &context.q));
        }
        RqPoly {
            coeffs: c,
            is_ntt_form: false,
            context: Some(context.clone()),
        }
    }
    #[test]
    fn test_ntt_constant_scalar() {
        let q = Scalar::new_modulus(18014398492704769u64);
        let context = Arc::new(RqPolyContext::new(8, &q));
        let v: Vec<u32> = vec![0; 8];

        let mut testpoly = from_vec(&v, context.clone());

        let tmp = Scalar::modulus(&Scalar::from(101 as u32), &q);
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
        let a = crate::randutils::sample_uniform_poly(arc.clone());
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

        let a = crate::randutils::sample_uniform_poly(arc.clone());
        let b = crate::randutils::sample_uniform_poly(arc.clone());
        let c = a.multiply(&b);
        let c1 = a.multiply_fast(&b);
        assert_eq!(c.coeffs, c1.coeffs);
    }

    #[test]
    fn test_find_root_scalar(){
        let context2 = RqPolyContext::new(4, &Scalar::new_modulus(12289));
        assert_eq!(context2.find_root().unwrap(), Scalar::from_u64_raw(8246u64));
    }

    #[test]
    fn test_lazy_ntt(){
        let q = Scalar::new_modulus(18014398492704769u64);
        let context = RqPolyContext::new(4, &q);
        let arc = Arc::new(context);
        let mut a = crate::randutils::sample_uniform_poly(arc.clone());
        let mut aa = a.clone();

        aa.forward_transform();
        a.lazy_ntt(); 

        // assert 
        assert_eq!(aa.coeffs, a.coeffs); 
    }


    #[test]
    fn test_lazy_inverse_ntt(){
        let q = Scalar::new_modulus(18014398492704769u64);
        let context = RqPolyContext::new(4, &q);
        let arc = Arc::new(context);
        let mut a = crate::randutils::sample_uniform_poly(arc.clone());
        a.set_ntt_form(true);
        let mut aa = a.clone();
        aa.inverse_transform();
        a.lazy_inverse_ntt(); 

        // assert 
        assert_eq!(aa.coeffs, a.coeffs); 
    }
}
