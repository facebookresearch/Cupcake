// use serde::{Serialize, Deserialize};
use crate::Serializable;
use std::convert::From;
use crate::integer_arith::scalar::Scalar;
use crate::rqpoly::RqPoly;
use std::convert::TryInto;

pub struct RqPolyWireModel<T> {
  pub coeffs: Vec<T>,
  pub is_ntt_form: bool,
}

pub struct ScalarWireModel{
  pub rep: u64,
}

impl From<Scalar> for ScalarWireModel {
  fn from(item: Scalar) -> Self {
    ScalarWireModel { rep: item.rep() }
  }
}


impl Serializable for Scalar{
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

impl<T> Serializable for RqPoly<T> where T: Serializable{
  fn to_bytes(&self) -> std::vec::Vec<u8> {
    let mut vec: Vec<u8> = vec![0,0];
    // push in the is ntt form.
    vec.push(self.is_ntt_form as u8);
    for i in 0..self.coeffs.len(){
      let mut bytes = self.coeffs[i].to_bytes();
      // append
      vec.append(& mut bytes);
    }
    vec
  }
  fn from_bytes(bytes: &std::vec::Vec<u8>) -> Self {
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

  // fn test_rqpoly_serialization(){
  //   let context = RqPolyContext::new(4, &Scalar::new_modulus(12289));
  //   let arc = Arc::new(context);
  //   let a = from_vec(&vec![1, 0, 0, 0], arc.clone());
  //   // initialize RqPolys
  //   // serialize
  //   // check get back the same poly
  // }
}



// impl<T> Serializable for RqPolyWireModel<T> where T: Serialize{

// }


// impl Serializable for ScalarWireModel{

// }
