// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use crate::integer_arith::scalar::Scalar;
use crate::rqpoly::RqPoly;
use crate::Serializable;
use crate::FVCiphertext;

#[cfg(test)]
use crate::FV;
#[cfg(test)]
use crate::integer_arith::ArithUtils;
#[cfg(test)]
use crate::traits::*;

use std::convert::TryInto;

impl Serializable for Scalar {
  fn to_bytes(&self) -> std::vec::Vec<u8> {
    let bytes = self.rep().to_be_bytes();
    let mut vec: Vec<u8> = vec![0; 8];
    vec.copy_from_slice(&bytes);
    vec
  }
  fn from_bytes(bytes: &std::vec::Vec<u8>) -> Self {
    let a: u64 = u64::from_be_bytes(bytes.as_slice().try_into().unwrap());
    Scalar::new(a)
  }
}

impl<T> Serializable for RqPoly<T>
where
  T: Serializable + Clone,
{
  fn to_bytes(&self) -> std::vec::Vec<u8> {
    let mut vec: Vec<u8> = vec![self.is_ntt_form as u8];
    for coeff in &self.coeffs {
      let mut bytes = coeff.to_bytes();
      vec.append(&mut bytes);
    }
    vec
  }
  fn from_bytes(bytes: &std::vec::Vec<u8>) -> Self {
    let mut coeffs = Vec::new();
    let is_ntt_form  = bytes[0] != 0;
    let mut  i : usize = 1;
    while i + 8 <= bytes.len() {
      coeffs.push(T::from_bytes(&bytes[i..i+8].to_vec()));
      i += 8;
    }
    RqPoly::new_without_context(&coeffs, is_ntt_form)
  }
}

impl<T> Serializable for FVCiphertext<T>
where
  T: Serializable + Clone,
{
  fn to_bytes(&self) -> std::vec::Vec<u8> {
    let mut ct0_bytes = self.0.to_bytes();
    let mut ct1_bytes = self.1.to_bytes();
    ct0_bytes.append(&mut ct1_bytes);
    ct0_bytes
  }
  fn from_bytes(bytes: &std::vec::Vec<u8>) -> Self {
    let twon = bytes.len();
    let n = twon / 2;
    (RqPoly::from_bytes(&bytes[0..n].to_vec()),RqPoly::from_bytes(&bytes[n..twon].to_vec()))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn test_scalar_serialization() {
    let c = Scalar::new(2);
    let bytes = c.to_bytes();
    let deserialized_c = Scalar::from_bytes(&bytes);
    assert_eq!(c, deserialized_c);
  }

  #[test]
  fn test_rqpoly_serialization() {
    let mut coeffs = Vec::<Scalar>::new();
    for i in 0..4 {
      coeffs.push(Scalar::from_u64_raw(i));
    }
    let testpoly = RqPoly::<Scalar>::new_without_context(&coeffs, false);
    let bytes = testpoly.to_bytes();
    let deserialized = RqPoly::<Scalar>::from_bytes(&bytes);
    assert_eq!(testpoly, deserialized);
  }

  #[test]
  fn test_fvciphertext_serialization(){
    // test ciphertext serialization
    let fv = FV::<Scalar>::default_2048();
    let (pk, _) = fv.generate_keypair();
    let mut v = vec![0; fv.n];
    for i in 0..fv.n {
        v[i] = i as u8;
    }
    // encrypt v
    let ct = fv.encrypt(&v, &pk);
    let bytes = ct.to_bytes();
    let ct_deserialized = FVCiphertext::from_bytes(&bytes);
    assert_eq!(ct_deserialized, ct);
  }
}
