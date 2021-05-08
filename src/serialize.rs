// use serde::{Serialize, Deserialize};
use crate::integer_arith::scalar::Scalar;
use crate::integer_arith::ArithUtils;
use crate::rqpoly::RqPoly;
use crate::FVCiphertext;
use crate::Serializable;
use std::convert::From;
use std::convert::TryInto;

pub struct ScalarWireModel {
  pub rep: u64,
}

impl From<Scalar> for ScalarWireModel {
  fn from(item: Scalar) -> Self {
    ScalarWireModel { rep: item.rep() }
  }
}

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
  T: Serializable,
{
  fn to_bytes(&self) -> std::vec::Vec<u8> {
    let mut vec: Vec<u8> = Vec::new();
    // push in the is ntt form.
    vec.push(self.is_ntt_form as u8);
    for i in 0..self.coeffs.len() {
      let mut bytes = self.coeffs[i].to_bytes();
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
    RqPoly::new(coeffs, is_ntt_form)
  }
}

impl<T> Serializable for FVCiphertext<T>
where
  T: Serializable,
{
  fn to_bytes(&self) -> std::vec::Vec<u8> {
    todo!()
  }
  fn from_bytes(_: &std::vec::Vec<u8>) -> Self {
    todo!()
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
    let testpoly = RqPoly::<Scalar>::new(coeffs, false);
    let bytes = testpoly.to_bytes();
    let deserialized = RqPoly::<Scalar>::from_bytes(&bytes);
    assert_eq!(testpoly, deserialized);
  }
}
