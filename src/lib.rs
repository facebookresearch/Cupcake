// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//! An implementation of the (additive homomorphism only) Fan-Vercauteren (FV) lattice-based homomorphic encryption scheme.
//! # Overview
//! Homomorphic encryption supports operations on encrypted data without knowing the decryption key.
//!
//! In order to use lattice-based homomorphic encryption, we first need to decide on the scheme to use and set up the parameters, including the polynomial degree (n) and the modulus (q).
//!
//! Currently, we only support one scheme (FV) and one set of parameters, corresponding to a polynomial degree of 2048 and a 54-bit prime modulus.
//! ```
//! let scheme = cupcake::default();
//! ```
//! # Setup
//! In order to encrypt and decrypt data, we needs to generate a keypair, i.e. a secret key and a public key.
//! ```
//! let scheme = cupcake::default();
//! use cupcake::traits::{KeyGeneration};
//! let (pk, sk) = scheme.generate_keypair();
//! ```
//! The public key can be used for encryption and the secret key can be used for encryption or decryption.
//!
//! # Encryption and Decryption
//!
//! The default plaintext space of Cupcake is `Vec<u8>` of fixed size n. We can encrypt a vector under a public key like so
//! ```
//! # let scheme = cupcake::default();
//! # use cupcake::traits::KeyGeneration;
//! # let (pk, sk) = scheme.generate_keypair();
//! use cupcake::traits::{SKEncryption, PKEncryption};
//! let v = vec![1; scheme.n];
//! let ct = scheme.encrypt(&v, &pk);
//! ```
//! Then, the ciphertext `ct` can be decrypted using the secret key:
//! ```
//! # let scheme = cupcake::default();
//! # use cupcake::traits::{KeyGeneration, SKEncryption, PKEncryption};
//! # let (pk, sk) = scheme.generate_keypair();
//! # let v = vec![1; scheme.n];
//! # let ct = scheme.encrypt(&v, &pk);
//! let w: Vec<u8>= scheme.decrypt(&ct, &sk);
//! assert_eq!(v, w);
//! ```
//! You may also use other plaintext types, which are vectors of `Scalar` of size n. Here `Scalar`  represents an integer modulo a fixed modulus t.
//! See the following example:
//! ```
//! # use cupcake::traits::{KeyGeneration, PKEncryption, SKEncryption};
//! use cupcake::integer_arith::scalar::Scalar;
//! let t = 199;
//! let scheme = cupcake::default_with_plaintext_mod(t);
//! let (pk, sk) = scheme.generate_keypair();
//! let plain_modulus = scheme.t.clone();
//! let pt = vec![Scalar::from(t-1 as u32); scheme.n];
//! let ct = scheme.encrypt(&pt, &pk);
//! let pt_actual: Vec<Scalar> = scheme.decrypt(&ct, &sk);
//! assert_eq!(pt_actual, pt);
//! ```
//! # Homomorphic Operations
//!
//! We can encrypt two vectors and add up the resulting ciphertexts.
//! ```
//! # let scheme = cupcake::default();
//! # use cupcake::traits::{KeyGeneration, SKEncryption, PKEncryption};
//! # let (pk, sk) = scheme.generate_keypair();
//! use cupcake::traits::{AdditiveHomomorphicScheme};
//! let z1 = vec![1; scheme.n];
//! let mut ctz1 = scheme.encrypt(&z1, &pk);
//! let z2 = vec![2; scheme.n];
//! let ctz2 = scheme.encrypt(&z2, &pk);
//! scheme.add_inplace(&mut ctz1, &ctz2);
//! // Now ctz1 should decrypt to vec![3; scheme.n];
//! let expected = vec![3; scheme.n];
//! let actual: Vec<u8> = scheme.decrypt(&ctz1, &sk);
//! assert_eq!(actual, expected);
//! ```
//! Alternatively, we can add a plaintext vector into a ciphertext
//! ```
//! # let scheme = cupcake::default();
//! # use cupcake::traits::{KeyGeneration, SKEncryption, PKEncryption};
//! # let (pk, sk) = scheme.generate_keypair();
//! use cupcake::traits::CipherPlainAddition;
//! let z = vec![1; scheme.n];
//! let mut ctz = scheme.encrypt(&z, &pk);
//! let p = vec![4; scheme.n];
//! scheme.add_plain_inplace(&mut ctz, &p);
//! // Now ctz should decrypt to vec![5; scheme.n]
//! let expected = vec![5; scheme.n];
//! let actual: Vec<u8> = scheme.decrypt(&ctz, &sk);
//! assert_eq!(actual, expected);
//! ```
//! # Rerandomization
//! Furthermore, you can rerandomize a ciphertext using the public key. The output is another ciphertext which will be still decrypt to the same plaintext, but cannot be linked to the input.
//! ```
//! # let scheme = cupcake::default();
//! # use cupcake::traits::{KeyGeneration, SKEncryption, PKEncryption};
//! # let (pk, sk) = scheme.generate_keypair();
//! # use cupcake::traits::{AdditiveHomomorphicScheme};
//! let mu = vec![1; scheme.n];
//! let mut ct = scheme.encrypt(&mu, &pk);
//! scheme.rerandomize(&mut ct, &pk);
//! // The new ct should still decrypt to mu.
//! let actual: Vec<u8> = scheme.decrypt(&ct, &sk);
//! let expected = mu;
//! assert_eq!(actual, expected);
//! ```
//! # Serialization
//! We provide methods to serialize  a ciphertext into a ```Vec<u8>```. Note that after deserialization, a proper context needs to be set before the further operations can be done on the ciphertext. See
//! the following example:
//! ```
//! # let scheme = cupcake::default();
//! # use cupcake::traits::{KeyGeneration, SKEncryption, PKEncryption};
//! # let (pk, sk) = scheme.generate_keypair();
//! # use cupcake::traits::{AdditiveHomomorphicScheme};
//! use crate::cupcake::traits::Serializable;
//! let v = vec![1; scheme.n];
//! let w = vec![1; scheme.n];
//! let ctv = scheme.encrypt(&v, &pk);
//! let ctw = scheme.encrypt(&w, &pk);
//! ```
//! We can call the `to_bytes` function to serialize.
//! ```
//! # let scheme = cupcake::default();
//! # use cupcake::traits::{KeyGeneration, SKEncryption, PKEncryption};
//! # let (pk, sk) = scheme.generate_keypair();
//! # use cupcake::traits::{AdditiveHomomorphicScheme};
//! # use crate::cupcake::traits::Serializable;
//! # let v = vec![1; scheme.n];
//! # let w = vec![1; scheme.n];
//! # let ctv = scheme.encrypt(&v, &pk);
//! # let ctw = scheme.encrypt(&w, &pk);
//! let ctv_serialized = ctv.to_bytes();
//! let ctw_serialized = ctw.to_bytes();
//! ```
//! In order to deserialize, use `scheme.from_bytes`.
//! ```
//! # let scheme = cupcake::default();
//! # use cupcake::traits::{KeyGeneration, SKEncryption, PKEncryption};
//! # let (pk, sk) = scheme.generate_keypair();
//! # use cupcake::traits::{AdditiveHomomorphicScheme};
//! # use crate::cupcake::traits::Serializable;
//! # let v = vec![1; scheme.n];
//! # let w = vec![1; scheme.n];
//! # let ctv = scheme.encrypt(&v, &pk);
//! # let ctw = scheme.encrypt(&w, &pk);
//! # let ctv_serialized = ctv.to_bytes();
//! # let ctw_serialized = ctw.to_bytes();
//! let mut ctv_deserialized = scheme.from_bytes(&ctv_serialized);
//! let ctw_deserialized = scheme.from_bytes(&ctw_serialized);
//! assert_eq!(ctv, ctv_deserialized);
//! ```
//! We can perform homomorphic operations on deserialized ciphertexts.
//! ```
//! # let scheme = cupcake::default();
//! # let (pk, sk) = scheme.generate_keypair();
//! # use crate::cupcake::traits::*;
//! # let v = vec![1; scheme.n];
//! # let w = vec![1; scheme.n];
//! # let ctv = scheme.encrypt(&v, &pk);
//! # let ctw = scheme.encrypt(&w, &pk);
//! # let ctv_serialized = ctv.to_bytes();
//! # let ctw_serialized = ctw.to_bytes();
//! # let mut ctv_deserialized = scheme.from_bytes(&ctv_serialized);
//! # let ctw_deserialized = scheme.from_bytes(&ctw_serialized);
//! scheme.add_inplace(&mut ctv_deserialized, &ctw_deserialized);
//! let expected = vec![2; scheme.n];
//! let actual: Vec<u8>= scheme.decrypt(&ctv_deserialized, &sk);
//! assert_eq!(actual, expected);
//! ```


pub mod integer_arith;
#[cfg(feature = "bench")]
pub mod rqpoly;
#[cfg(not(feature = "bench"))]
mod rqpoly;
pub mod traits;
mod serialize;
mod utils;
#[cfg(feature = "bench")]
pub mod randutils;
#[cfg(not(feature = "bench"))]
mod randutils;

use integer_arith::scalar::Scalar;
use integer_arith::{SuperTrait, ArithUtils};
use traits::*;
use std::sync::Arc;

/// Plaintext type
pub type FVPlaintext<T> = Vec<T>;
/// Default plaintext type
pub type DefaultFVPlaintext = Vec<u8>;
/// Ciphertext type
pub type FVCiphertext<T> = (RqPoly<T>, RqPoly<T>);

/// Default scheme type
pub type DefaultShemeType = FV<Scalar>;

/// SecretKey type
pub struct SecretKey<T>(RqPoly<T>);
use rqpoly::{FiniteRingElt, RqPoly, RqPolyContext};

pub fn default() -> DefaultShemeType {
    FV::<Scalar>::default_2048()
}

pub fn default_with_plaintext_mod(t: u32) -> DefaultShemeType {
    FV::<Scalar>::default_2048_with_plaintext_mod(t)
}

/// (Additive only version of) the Fan-Vercauteren homomoprhic encryption scheme.
pub struct FV<T>
where
    T: ArithUtils<T>,
{
    pub n: usize,
    pub t: T,
    pub q: T,
    pub delta: T,
    pub stdev: f64,
    pub qdivtwo: T,
    pub flooding_stdev: f64,
    context: Arc<RqPolyContext<T>>,
    poly_multiplier: fn(&RqPoly<T>, &RqPoly<T>) -> RqPoly<T>,
}

impl<T> FV<T>
where
T: ArithUtils<T>{
    fn convert_pt_u8_to_scalar(&self, pt: &DefaultFVPlaintext) -> FVPlaintext<T>{
        if T::to_u64(&self.t) != 256u64{
            panic!("plaintext modulus should be 256")
        }
        let mut pt1 = vec![];
        for pt_coeff in pt.iter(){
            pt1.push(T::from_u32(*pt_coeff as u32,  &self.t));
        }
        pt1
    }

    fn convert_pt_scalar_to_u8(&self, pt: FVPlaintext<T>) -> DefaultFVPlaintext{
        if T::to_u64(&self.t) != 256u64{
            panic!("plaintext modulus should be 256")
        }
        let mut pt1 = vec![];
        for pt_coeff in pt.iter(){
            pt1.push(T::to_u64(pt_coeff) as u8);
        }
        pt1
    }
}

impl<T> CipherPlainAddition<FVCiphertext<T>, FVPlaintext<T>> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq,
{
    // add a plaintext into a FVCiphertext.
    fn add_plain_inplace(&self, ct: &mut FVCiphertext<T>, pt: &FVPlaintext<T>) {
        for (ct_coeff, pt_coeff) in ct.1.coeffs.iter_mut().zip(pt.iter()) {
            let temp = T::mul(&pt_coeff, &self.delta);
            *ct_coeff = T::add_mod(ct_coeff, &temp, &self.q);
        }
    }
}


impl<T> CipherPlainAddition<FVCiphertext<T>, DefaultFVPlaintext> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq + From<u32>,
{
    // add a plaintext into a FVCiphertext.
    fn add_plain_inplace(&self, ct: &mut FVCiphertext<T>, pt: &DefaultFVPlaintext) {
        for (ct_coeff, pt_coeff) in ct.1.coeffs.iter_mut().zip(pt.iter()) {
            let temp = T::mul(&T::from(*pt_coeff as u32), &self.delta);
            *ct_coeff = T::add_mod(ct_coeff, &temp, &self.q);
        }
    }
}


impl<T> AdditiveHomomorphicScheme<FVCiphertext<T>, SecretKey<T>> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq +  From<u32>,
{
    fn add_inplace(&self, ct1: &mut FVCiphertext<T>, ct2: &FVCiphertext<T>) {
        ct1.0.add_inplace(&ct2.0);
        ct1.1.add_inplace(&ct2.1);
    }

    // rerandomize a ciphertext
    fn rerandomize(&self, ct: &mut FVCiphertext<T>, pk: &FVCiphertext<T>) {
        // add a public key encryption of zero.
        let c_mask = self.encrypt_zero(pk);
        self.add_inplace(ct, &c_mask);

        // add large noise poly for noise flooding.
        let elarge =
            rqpoly::randutils::sample_gaussian_poly(self.context.clone(), self.flooding_stdev);
        ct.1.add_inplace(&elarge);
    }
}

// constructor and random poly sampling
impl<T> FV<T>
where
    T: SuperTrait<T>+ PartialEq + Serializable,
    RqPoly<T>: FiniteRingElt + NTT<T>,
{
    pub fn new(n: usize, q: &T) -> Self {
        Self::new_with_ptxt_mod(n, &T::new_modulus(256), q)
    }

    pub fn new_with_ptxt_mod(n: usize, t: &T, q: &T) -> Self {
        let context = Arc::new(RqPolyContext::new(n, q));
        type RqPolyMultiplier<T> = fn(&RqPoly<T>, &RqPoly<T>) -> RqPoly<T>;
        let default_multiplier: RqPolyMultiplier<T>;
        if context.is_ntt_enabled {
            default_multiplier =
                |op1: &RqPoly<T>, op2: &RqPoly<T>| -> RqPoly<T> { op1.multiply_fast(op2) };
        } else {
            default_multiplier =
                |op1: &RqPoly<T>, op2: &RqPoly<T>| -> RqPoly<T> { op1.multiply(op2) };
        }
        FV {
            n,
            t: t.clone(),
            flooding_stdev: 2f64.powi(40),
            delta: T::div(q, &t), // &q/t,
            qdivtwo: T::div(q, &T::from(2 as u32)), // &q/2,
            q: q.clone(),
            stdev: 3.2,
            context,
            poly_multiplier: default_multiplier,
        }
    }

    pub fn from_bytes(&self, bytes: &Vec<u8>) -> FVCiphertext<T>{
        let mut ct = FVCiphertext::<T>::from_bytes(bytes);
        self.set_context(&mut ct);
        ct
    }

    fn set_context(&self, ctxt: &mut FVCiphertext<T>){
        ctxt.0.set_context(self.context.clone());
        ctxt.1.set_context(self.context.clone());
    }
}

impl FV<Scalar> {
    /// Construct a scheme with default parameters and plaintext modulus 256.
    pub fn default_2048() -> FV<Scalar> {
        let q = Scalar::new_modulus(18014398492704769u64);
        Self::new(2048, &q)
    }

    /// Construct a scheme with provided plaintext modulus.
    pub fn default_2048_with_plaintext_mod(t: u32) -> FV<Scalar> {
        if t > 2u32.pow(10){
            panic!("plain text modulus should not be more than 10 bits.")
        }
        let q = Scalar::new_modulus(18014398492704769u64);
        let t = Scalar::new_modulus(t as u64);
        Self::new_with_ptxt_mod(2048, &t, &q)
    }
}

#[cfg(feature = "bigint")]
impl FV<BigInt> {
    pub fn default_2048() -> FV<BigInt> {
        let q = BigInt::from_hex("3fffffff000001");
        let context = Arc::new(RqPolyContext::new(2048, &q));
        let multiplier = |op1: &RqPoly<BigInt>, op2: &RqPoly<BigInt>| -> RqPoly<BigInt> {
            op1.multiply_fast(op2)
        };

        FV {
            n: 2048,
            q: q.clone(),
            delta: &q / 256,
            qdivtwo: &q / 2,
            stdev: 3.2,
            flooding_stdev: 1e40_f64,
            context: context,
            poly_multiplier: multiplier,
        }
    }
}


impl<T> KeyGeneration<FVCiphertext<T>,  SecretKey<T>> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq  + From<u32>,
{
    fn generate_key(&self) -> SecretKey<T> {
        let mut skpoly = rqpoly::randutils::sample_ternary_poly(self.context.clone());
        if self.context.is_ntt_enabled {
            skpoly.forward_transform();
        }
        SecretKey(skpoly)
    }

    fn generate_keypair(&self) -> (FVCiphertext<T>, SecretKey<T>) {
        let sk = self.generate_key();
        let mut pk = self.encrypt_zero_sk(&sk);
        if self.context.is_ntt_enabled {
            pk.0.forward_transform();
            pk.1.forward_transform();
        }
        (pk, sk)
    }
}

impl<T> EncryptionOfZeros<FVCiphertext<T>,  SecretKey<T>> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq  + From<u32>,
{
    fn encrypt_zero(&self, pk: &FVCiphertext<T>) -> FVCiphertext<T> {
        let mut u = rqpoly::randutils::sample_ternary_poly_prng(self.context.clone());
        let e1 = rqpoly::randutils::sample_gaussian_poly(self.context.clone(), self.stdev);
        let e2 = rqpoly::randutils::sample_gaussian_poly(self.context.clone(), self.stdev);

        if self.context.is_ntt_enabled {
            u.forward_transform();
        }
        // c0 = au + e1
        let mut c0 = (self.poly_multiplier)(&pk.0, &u);
        c0.add_inplace(&e1);

        // c1 = bu + e2
        let mut c1 = (self.poly_multiplier)(&pk.1, &u);
        c1.add_inplace(&e2);

        (c0, c1)
    }

    fn encrypt_zero_sk(&self, sk: &SecretKey<T>) -> FVCiphertext<T> {
        let e = rqpoly::randutils::sample_gaussian_poly(self.context.clone(), self.stdev);
        let a = rqpoly::randutils::sample_uniform_poly(self.context.clone());
        let mut b = (self.poly_multiplier)(&a, &sk.0);
        b.add_inplace(&e);
        (a, b)
    }
}

impl<T> PKEncryption<FVCiphertext<T>, FVPlaintext<T>, SecretKey<T>> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq  + From<u32>,
{
    fn encrypt(&self, pt: &FVPlaintext<T>, pk: &FVCiphertext<T>) -> FVCiphertext<T> {
        // use public key to encrypt
        // pk = (a, as+e) = (a,b)
        let (c0, mut c1) = self.encrypt_zero(pk);
        // c1 = bu+e2 + Delta*m
        let iter = c1.coeffs.iter_mut().zip(pt.iter());
        for (x, y) in iter {
            let temp = T::mul(&y, &self.delta);
            *x = T::add_mod(x, &temp, &self.q);
        }
        (c0, c1)
    }
}

impl<T> PKEncryption<FVCiphertext<T>, DefaultFVPlaintext, SecretKey<T>> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq + From<u32>,
{
    fn encrypt(&self, pt: &DefaultFVPlaintext, pk: &FVCiphertext<T>) -> FVCiphertext<T> {
        let pt1 = self.convert_pt_u8_to_scalar(pt);
        self.encrypt(&pt1, pk)
    }
}

impl<T> SKEncryption<FVCiphertext<T>, DefaultFVPlaintext, SecretKey<T>> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq + From<u32>,
{
    fn encrypt_sk(&self, pt: &DefaultFVPlaintext, sk: &SecretKey<T>) -> FVCiphertext<T>
        {
            let pt1 = self.convert_pt_u8_to_scalar(pt);
            self.encrypt_sk(&pt1, sk)
        }

    fn decrypt(&self, ct: &FVCiphertext<T>, sk: &SecretKey<T>) -> DefaultFVPlaintext{
        let pt1 = self.decrypt(ct, sk);
        self.convert_pt_scalar_to_u8(pt1)
    }
}

// This implements the sk-encryption for BFV scheme.
impl<T> SKEncryption<FVCiphertext<T>, FVPlaintext<T>, SecretKey<T>> for FV<T>
where
    RqPoly<T>: FiniteRingElt,
    T: Clone + ArithUtils<T> + PartialEq + From<u32>,
{
    fn encrypt_sk(&self, pt: &FVPlaintext<T>, sk: &SecretKey<T>) -> FVCiphertext<T> {
        let e = rqpoly::randutils::sample_gaussian_poly(self.context.clone(), self.stdev);
        let a = rqpoly::randutils::sample_uniform_poly(self.context.clone());

        let mut b = (self.poly_multiplier)(&a, &sk.0);
        b.add_inplace(&e);

        // add scaled plaintext to
        let iter = b.coeffs.iter_mut().zip(pt.iter());
        for (x, y) in iter {
            let temp = T::mul(&y, &self.delta);
            *x = T::add_mod(x, &temp, &self.q);
        }
        (a, b)
    }

    fn decrypt(&self, ct: &FVCiphertext<T>, sk: &SecretKey<T>) -> FVPlaintext<T> {
        let temp1 = (self.poly_multiplier)(&ct.0, &sk.0);
        let mut phase = ct.1.clone();
        phase.sub_inplace(&temp1);
        // then, extract value from phase.
        let mut c: Vec<T> = vec![];
        for x in phase.coeffs {
            // let mut tmp = x << 8;  // x * t, need to make sure there's no overflow.
            let mut tmp = T::mul(&x, &self.t);
            // tmp += &self.qdivtwo;
            tmp = T::add(&tmp, &self.qdivtwo);
            // tmp /= &self.q;
            tmp = T::div(&tmp, &self.q);
            // modulo t.
            tmp = T::modulus(&tmp, &self.t);

            c.push(tmp);
        }
        c
    }
}

#[cfg(test)]
mod fv_scalar_tests {
    use super::*;
    #[test]
    fn test_sk_encrypt_toy_param_scalar() {
        let fv = FV::new(16, &Scalar::new_modulus(65537));

        let sk = fv.generate_key();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        let ct = fv.encrypt_sk(&v, &sk);

        let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);

        assert_eq!(v, pt_actual);
    }

    #[test]
    fn test_sk_encrypt_scalar() {
        let fv = FV::<Scalar>::default_2048();

        let sk = fv.generate_key();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        let ct = fv.encrypt_sk(&v, &sk);

        let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);

        assert_eq!(v, pt_actual);
    }

    #[test]
    fn test_encrypt_default_param_scalar() {
        let fv = FV::<Scalar>::default_2048();

        let (pk, sk) = fv.generate_keypair();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        let ct = fv.encrypt(&v, &pk);

        let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);

        assert_eq!(v, pt_actual);
    }

    #[test]
    fn test_rerandomize_scalar() {
        let fv = FV::<Scalar>::default_2048();

        let (pk, sk) = fv.generate_keypair();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        let mut ct = fv.encrypt(&v, &pk);

        fv.rerandomize(&mut ct, &pk);

        let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);

        assert_eq!(v, pt_actual);
    }

    #[test]
    fn test_add_scalar() {
        let fv = FV::<Scalar>::default_2048();
        let (pk, sk) = fv.generate_keypair();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }

        let mut w: Vec<u8> = vec![];
        for i in 0..fv.n {
            w.push((fv.n - i) as u8);
        }

        let mut vplusw = vec![];
        for _ in 0..fv.n {
            vplusw.push(fv.n as u8);
        }
        // encrypt v
        let mut ctv = fv.encrypt_sk(&v, &sk);
        let ctw = fv.encrypt(&w, &pk);

        // ct_v + ct_w.
        fv.add_inplace(&mut ctv, &ctw);
        let pt_after_add: DefaultFVPlaintext = fv.decrypt(&ctv, &sk);
        assert_eq!(pt_after_add, vplusw);
    }

    #[test]
    fn test_add_plain_scalar() {
        let fv = FV::<Scalar>::default_2048();
        let (pk, sk) = fv.generate_keypair();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }

        let mut w: Vec<u8> = vec![];
        for i in 0..fv.n {
            w.push((fv.n - i) as u8);
        }

        let mut vplusw = vec![];
        for _ in 0..fv.n {
            vplusw.push(fv.n as u8);
        }
        // encrypt v
        let mut ct = fv.encrypt(&v, &pk);

        // ct_v + w.
        fv.add_plain_inplace(&mut ct, &w);

        let pt_after_add: DefaultFVPlaintext = fv.decrypt(&ct, &sk);

        assert_eq!(pt_after_add, vplusw);
    }

    #[test]
    fn test_flexible_plaintext_encrypt() {
        let t = 199;
        let fv = crate::default_with_plaintext_mod(t);
        let (pk, sk) = fv.generate_keypair();
        let plain_modulus = fv.t.clone();
        let pt = vec![Scalar::from_u32(t-1, &plain_modulus); fv.n];
        let ct = fv.encrypt(&pt, &pk);
        let pt_actual: Vec<Scalar> = fv.decrypt(&ct, &sk);
        assert_eq!(pt_actual, pt);
    }

    #[test]
    fn test_flexible_plaintext_addition() {
        let t = 199;
        let fv = crate::default_with_plaintext_mod(t);
        let (pk, sk) = fv.generate_keypair();
        let plain_modulus = fv.t.clone();
        let v = vec![Scalar::from_u32(t-1, &plain_modulus); fv.n];
        let w = vec![Scalar::from_u32(t-1, &plain_modulus); fv.n];
        let v_plus_w = vec![Scalar::from_u32(t-2, &plain_modulus); fv.n];
        let mut ctv = fv.encrypt(&v, &pk);
        let ctw = fv.encrypt(&w, &pk);
        fv.add_inplace(&mut ctv, &ctw);
        let pt_actual: Vec<Scalar>= fv.decrypt(&ctv, &sk);
        assert_eq!(pt_actual, v_plus_w);
    }

    #[test]
    fn test_flexible_plaintext_add_plaintext() {
        let t = 199;
        let fv = crate::default_with_plaintext_mod(t);
        let (pk, sk) = fv.generate_keypair();
        let plain_modulus = fv.t.clone();
        let v = vec![Scalar::from_u32(t-1, &plain_modulus); fv.n];
        let w = vec![Scalar::from_u32(1, &plain_modulus); fv.n];
        let v_plus_w = vec![Scalar::from_u32(0, &plain_modulus); fv.n];
        let mut ct = fv.encrypt(&v, &pk);
        fv.add_plain_inplace(&mut ct, &w);
        let pt_actual: Vec<Scalar> = fv.decrypt(&ct, &sk);
        assert_eq!(pt_actual, v_plus_w);
    }
}

// unit tests.
#[cfg(feature = "bigint")]
#[cfg(test)]
mod fv_bigint_tests {
    use super::*;
    #[test]
    fn test_sk_encrypt() {
        let fv = FV::new(16, &BigInt::from(12289));

        let sk = fv.generate_key();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        let ct = fv.encrypt_sk(&v, &sk);

        let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);

        assert_eq!(v, pt_actual);
    }

    #[test]
    fn test_encrypt_toy_param() {
        let fv = FV::new(4, &BigInt::from(65537));

        let (pk, sk) = fv.generate_keypair();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        for _ in 0..10 {
            let ct = fv.encrypt(&v, &pk);
            let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);
            assert_eq!(v, pt_actual);
        }
    }

    #[test]
    fn test_encrypt_nonntt_toy_param() {
        let fv = FV::new(4, &BigInt::from(1000000));

        let (pk, sk) = fv.generate_keypair();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        for _ in 0..10 {
            let ct = fv.encrypt(&v, &pk);
            let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);
            assert_eq!(v, pt_actual);
        }
    }

    #[test]
    fn test_encrypt_large_param() {
        let fv = FV::<BigInt>::default_2048();

        let (pk, sk) = fv.generate_keypair();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        let ct = fv.encrypt(&v, &pk);

        let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);

        assert_eq!(v, pt_actual);
    }

    #[test]
    fn test_rerandomize() {
        let fv = FV::<BigInt>::default_2048();

        let (pk, sk) = fv.generate_keypair();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }
        let mut ct = fv.encrypt(&v, &pk);

        fv.rerandomize(&mut ct, &pk);

        let pt_actual: Vec<u8> = fv.decrypt(&ct, &sk);

        assert_eq!(v, pt_actual);
    }
    #[test]
    fn test_add() {
        let fv = FV::new(16, &BigInt::from(12289));

        let sk = fv.generate_key();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }

        let mut w: Vec<u8> = vec![];
        for i in 0..fv.n {
            w.push((fv.n - i) as u8);
        }

        let mut vplusw = vec![];
        for _ in 0..fv.n {
            vplusw.push(fv.n as u8);
        }
        // encrypt v
        let mut ctv = fv.encrypt_sk(&v, &sk);
        let ctw = fv.encrypt_sk(&w, &sk);

        // ct_v + ct_w.
        fv.add_inplace(&mut ctv, &ctw);

        let pt_after_add = fv.decrypt(&ctv, &sk);

        assert_eq!(pt_after_add, vplusw);
    }

    #[test]
    fn test_add_plain() {
        let fv = FV::new(16, &BigInt::from(12289));
        let sk = fv.generate_key();

        let mut v = vec![0; fv.n];
        for i in 0..fv.n {
            v[i] = i as u8;
        }

        let mut w: Vec<u8> = vec![];
        for i in 0..fv.n {
            w.push((fv.n - i) as u8);
        }

        let mut vplusw = vec![];
        for _ in 0..fv.n {
            vplusw.push(fv.n as u8);
        }
        // encrypt v
        let mut ct = fv.encrypt_sk(&v, &sk);

        // ct_v + w.
        fv.add_plain_inplace(&mut ct, &w);

        let pt_after_add = fv.decrypt(&ct, &sk);

        assert_eq!(pt_after_add, vplusw);
    }
}
