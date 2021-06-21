// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use crate::integer_arith::{ArithOperators, ArithUtils, SuperTrait};
use modinverse::modinverse;
use rand::rngs::StdRng;
use rand::FromEntropy;
use rand::RngCore;
use ::std::ops;
pub use std::sync::Arc;

/// The ScalarContext class contains useful auxilliary information for fast modular reduction against a Scalar instance.
#[derive(Debug, PartialEq, Eq, Clone)]
struct ScalarContext {
    barrett_ratio: (u64, u64),
}

impl ScalarContext {
    fn new(q: u64) -> Self {
        let ratio = Self::compute_barrett_ratio(q);
        ScalarContext {
            barrett_ratio: ratio,
        }
    }

    /// Compute floor(2^128/q) and put it in 2 u64s as (low-word, high-word)
    fn compute_barrett_ratio(q: u64) -> (u64, u64) {
        // 2^127 = s*q + t.
        let a = 1u128 << 127;
        let mut t = a % (q as u128);
        let mut s = (a - t) / (q as u128);

        s <<= 1;
        t <<= 1;
        if t >= (q as u128) {
            s += 1;
        }
        (s as u64, (s >> 64) as u64)
    }
}

/// The Scalar struct is a wrapper around u64 which has optional fast modular arithmetic through ScalarContext.
#[derive(Debug, Clone)]
pub struct Scalar {
    context: Option<ScalarContext>,
    rep: u64,
    bit_count: usize,
}

impl Scalar {
    /// Construct a new scalar from u64.
    pub fn new(a: u64) -> Self {
        Scalar {
            rep: a,
            context: None,
            bit_count: 0,
        }
    }

    pub fn rep(&self) -> u64{
        self.rep
    }
}

/// Trait implementations
impl SuperTrait<Scalar> for Scalar {}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.rep == other.rep
    }
}

// Conversions
impl From<u32> for Scalar {
    fn from(item: u32) -> Self {
        Scalar {  context: None, rep: item as u64, bit_count: 0 }
    }
}

impl From<u64> for Scalar {
    fn from(item: u64) -> Self {
        Scalar {  context: None, rep: item, bit_count: 0 }
    }
}

impl From<Scalar> for u64{
    fn from(item: Scalar) -> u64 {
        item.rep
    }
}

// Operators
impl ops::Add<&Scalar> for Scalar {
    type Output = Scalar;
    fn add(self, v: &Scalar) -> Scalar {
        Scalar::new(self.rep + v.rep)
    }
}

impl ops::Add<Scalar> for Scalar {
    type Output = Scalar;
    fn add(self, v: Scalar) -> Scalar {
        self + &v
    }
}

impl ops::Sub<&Scalar> for Scalar {
    type Output = Scalar;
    fn sub(self, v: &Scalar) -> Scalar {
         Scalar::new(self.rep - v.rep)
    }
}

impl ops::Sub<Scalar> for Scalar {
    type Output = Scalar;
    fn sub(self, v: Scalar) -> Scalar {
        self - &v
    }
}

impl ops::Mul<u64> for Scalar {
    type Output = Scalar;
    fn mul(self, v: u64) -> Scalar {
        Scalar::new(self.rep * v)
    }
}

impl ArithOperators for Scalar{
    fn add_u64(&mut self, a: u64){
        self.rep += a;
    }

    fn sub_u64(&mut self, a: u64){
        self.rep -= a;
    }

    fn rep(&self) -> u64{
        self.rep
    }
}



// Trait implementation
impl ArithUtils<Scalar> for Scalar {
    fn new_modulus(q: u64) -> Scalar {
        Scalar {
            rep: q,
            context: Some(ScalarContext::new(q)),
            bit_count: 64 - q.leading_zeros() as usize,
        }
    }

    fn sub(a: &Scalar, b: &Scalar) -> Scalar {
        Scalar::new(a.rep - b.rep)
    }

    fn div(a: &Scalar, b: &Scalar) -> Scalar {
        Scalar::new(a.rep / b.rep)
    }

    fn add_mod(a: &Scalar, b: &Scalar, q: &Scalar) -> Scalar {
        let mut sum = a.rep + b.rep;
        if sum >= q.rep {
            sum -= q.rep;
        }
        Scalar::new(sum)
    }

    fn sub_mod(a: &Scalar, b: &Scalar, q: &Scalar) -> Scalar {
        Scalar::_sub_mod(a, b, q.rep)
    }

    fn mul_mod(a: &Scalar, b: &Scalar, q: &Scalar) -> Scalar {
        let res = Scalar::_barret_multiply(a, b, q.context.as_ref().unwrap().barrett_ratio, q.rep);
        Scalar::new(res)
    }

    fn inv_mod(a: &Scalar, q: &Scalar) -> Scalar {
        Scalar::_inv_mod(a, q.rep)
    }

    fn from_u32(a: u32, q: &Scalar) -> Scalar {
        Scalar::new((a as u64) % q.rep)
    }

    fn from_u32_raw(a: u32) -> Scalar {
        Scalar::new(a as u64)
    }

    fn from_u64_raw(a: u64) -> Scalar {
        Scalar::new(a)
    }

    fn pow_mod(base: &Scalar, b: &Scalar, q: &Scalar) -> Scalar {
        let bits: Vec<bool> = b.get_bits();
        let mut res = Self::one();
        res = Self::modulus(&res, q);
        let mut pow = Scalar::new(base.rep);
        for bit in bits.iter() {
            if *bit {
                res = Self::mul_mod(&res, &pow, q);
            }
            pow = Self::mul_mod(&pow, &pow, q);
        }
        res
    }

    fn double(a: &Scalar) -> Scalar {
        Scalar::new(a.rep << 1)
    }

    fn sample_blw(upper_bound: &Scalar) -> Scalar {
        loop {
            let n = Self::_sample(upper_bound.bit_count);
            if n < upper_bound.rep {
                return Scalar::new(n);
            }
        }
    }

    // sample below using a given rng.
    fn sample_below_from_rng(upper_bound: &Scalar, rng: &mut StdRng) -> Self {
        loop {
            let n = Self::_sample_form_rng(upper_bound.bit_count, rng);
            if n < upper_bound.rep {
                return Scalar::new(n);
            }
        }
    }

    fn modulus(a: &Scalar, q: &Scalar) -> Scalar {
        Scalar::new(a.rep % q.rep)
    }

    fn mul(a: &Scalar, b: &Scalar) -> Scalar {
        Scalar::new(a.rep * b.rep)
    }

    fn to_u64(a: &Scalar) -> u64 {
        a.rep
    }

    fn add(a: &Scalar, b: &Scalar) -> Scalar {
        Scalar::new(a.rep + b.rep)
    }
}

impl Scalar {
    /// Bit length of this scalar.
    fn bit_length(&self) -> usize {
        64 - self.rep.leading_zeros() as usize
    }

    /// Return a vector of booleans representing the bits of this scalar, starting from the least significant bit.
    fn get_bits(&self) -> Vec<bool> {
        let len = self.bit_length();
        let mut res = vec![];
        let mut mask = 1u64;
        for _ in 0..len {
            res.push((self.rep & mask) != 0);
            mask <<= 1;
        }
        res
    }

    fn _sample_form_rng(bit_size: usize, rng: &mut StdRng) -> u64 {
        let bytes = (bit_size - 1) / 8 + 1;
        let mut buf: Vec<u8> = vec![0; bytes];
        rng.fill_bytes(&mut buf);

        // from vector to u64.
        let mut a = 0u64;
        for x in buf.iter() {
            a <<= 8;
            a += *x as u64;
        }
        a >>= bytes * 8 - bit_size;
        a
    }

    fn _sample(bit_size: usize) -> u64 {
        let mut rng = StdRng::from_entropy();
        Self::_sample_form_rng(bit_size, &mut rng)
    }

    fn _sub_mod(a: &Scalar, b: &Scalar, q: u64) -> Self {
        let diff;
        if a.rep >= b.rep {
            diff = a.rep - b.rep;
        } else {
            diff = a.rep + q - b.rep;
        }
        Scalar::new(diff)
    }

    fn _slowmul_mod(a: &Scalar, b: &Scalar, q: u64) -> Self {
        let res = (a.rep as u128) * (b.rep as u128);
        Scalar::new((res % (q as u128)) as u64)
    }

    fn _multiply_u64(a: u64, b: u64) -> (u64, u64) {
        let res = (a as u128) * (b as u128);
        (res as u64, (res >> 64) as u64)
    }

    fn _add_u64(a: u64, b: u64) -> (u64, bool) {
        let res = (a as u128 + b as u128) as u64;
        (res, res < a)
    }

    fn _barret_reduce(a: (u64, u64), ratio: (u64, u64), q: u64) -> u64 {
        // compute w = a*ratio >> 128.

        // start with lw(a1r1)
        // let mut w= Scalar::multiply_u64(a.1, ratio.1).0;
        let mut w = a.1.wrapping_mul(ratio.1);

        let a0r0 = Scalar::_multiply_u64(a.0, ratio.0);

        let a0r1 = Scalar::_multiply_u64(a.0, ratio.1);

        // w += hw(a0r1)
        w += a0r1.1;

        // compute hw(a0r0) + lw(a0r1), add carry into w. put result into tmp.
        let (tmp, carry) = Scalar::_add_u64(a0r0.1, a0r1.0);
        w += carry as u64;

        // Round2
        let a1r0 = Scalar::_multiply_u64(a.1, ratio.0);
        w += a1r0.1;
        // final carry
        let (_, carry2) = Scalar::_add_u64(a1r0.0, tmp);
        w += carry2 as u64;

        // low = w*q mod 2^64.
        // let low = Scalar::multiply_u64(w, q).0;
        let low = w.wrapping_mul(q);

        let mut res;
        if a.0 >= low {
            res = a.0 - low;
        } else {
            // res = a.0 + 2^64 - low.
            res = a.0 + (!low) + 1;
        }

        if res >= q {
            res -= q;
        }
        res
    }

    fn _inv_mod(a: &Scalar, q: u64) -> Self {
        Scalar::new(modinverse(a.rep as i128, q as i128).unwrap() as u64)
    }

    fn _barret_multiply(a: &Scalar, b: &Scalar, ratio: (u64, u64), q: u64) -> u64 {
        let prod = Scalar::_multiply_u64(a.rep, b.rep);
        Scalar::_barret_reduce(prod, ratio, q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bitlength() {
        assert_eq!(Scalar::from(2u32).bit_length(), 2);
        assert_eq!(Scalar::from(16u32).bit_length(), 5);
        assert_eq!(Scalar::from_u64_raw(18014398492704769u64).bit_length(), 54);
    }

    #[test]
    fn test_getbits() {
        assert_eq!(Scalar::from(1u32).get_bits(), vec![true]);
        assert_eq!(Scalar::from(2u32).get_bits(), vec![false, true]);
        assert_eq!(Scalar::from(5u32).get_bits(), vec![true, false, true]);
        assert_eq!(
            Scalar::from_u64_raw(127).get_bits(),
            vec![true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_sample_bitsize() {
        let bit_size = 54;
        let bound = 1u64 << bit_size;
        for _ in 0..10 {
            let a = Scalar::_sample(bit_size);
            assert!(a < bound);
        }
    }

    #[test]
    fn test_sample_below() {
        let q: u64 = 18014398492704769;
        let q_scalar = Scalar::new_modulus(q);
        for _ in 0..10 {
            assert!(Scalar::sample_blw(&q_scalar).rep < q);
        }
    }
    #[test]
    fn test_equality() {
        assert_eq!(Scalar::zero(), Scalar::zero());
    }

    #[test]
    fn test_subtraction() {
        let a = Scalar::zero();
        let b = Scalar::one();
        let c = Scalar::_sub_mod(&a, &b, 12289);
        assert_eq!(c.rep, 12288);
    }

    #[test]
    fn test_inverse() {
        let q = Scalar::new(11);
        let c = Scalar::new(2);
        let a = Scalar::inv_mod(&c, &q);
        assert_eq!(a.rep, 6);
    }

    #[test]
    fn test_mul_mod() {
        let q = 11u64;
        let c = Scalar::new(4);
        let a = Scalar::_slowmul_mod(&c, &c, q);
        assert_eq!(a.rep, 5);
    }

    #[test]
    fn test_pow_mod() {
        let q = Scalar::new_modulus(11);
        let c = Scalar::new(4);
        let a = Scalar::pow_mod(&c, &c, &q);
        assert_eq!(a.rep, 3);
    }

    #[test]
    fn test_pow_mod_large() {
        let q = Scalar::new_modulus(12289);
        let two = Scalar::new(2);
        let mut a: Scalar = Scalar::from_u64_raw(3);
        a = Scalar::modulus(&a, &q);

        for _ in 0..10 {
            a = Scalar::pow_mod(&a, &two, &q);
            assert!(a.rep < q.rep);
        }
    }

    #[test]
    fn test_barret_ratio() {
        let q = 18014398492704769u64;
        assert_eq!(
            ScalarContext::compute_barrett_ratio(q),
            (17592185012223u64, 1024u64)
        );
    }

    #[test]
    fn test_barret_reduction() {
        let q = 18014398492704769;
        let ratio = (17592185012223u64, 1024u64);

        let a: (u64, u64) = (1, 0);
        let b = Scalar::_barret_reduce(a, ratio, q);
        assert_eq!(b, 1);

        let a: (u64, u64) = (q, 0);
        let b = Scalar::_barret_reduce(a, ratio, q);
        assert_eq!(b, 0);

        let a: (u64, u64) = (0, 1);
        let b = Scalar::_barret_reduce(a, ratio, q);
        assert_eq!(b, 17179868160);
    }

    #[test]
    fn test_barret_multiply() {
        let q: u64 = 18014398492704769;
        let ratio = (17592185012223u64, 1024u64);

        let a = Scalar::new(q - 2);
        let b = Scalar::new(q - 3);
        let c = Scalar::_barret_multiply(&a, &b, ratio, q);

        assert_eq!(c, 6);
    }

    #[test]
    fn test_operator_add(){
        let a = Scalar::new(123);
        let b = Scalar::new(123);
        let c = a + &b;
        assert_eq!(u64::from(c), 246u64);
    }
}
