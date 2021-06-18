// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::scalar::Scalar;
use super::ArithUtils;



fn butterfly<T>(X: &mut T, Y: &mut T, W: &T, q: &T) where T: ArithUtils<T>{
  let temp  = T::mul_mod(Y, W, q);
  *Y = T::sub_mod(X, &temp, q);
  *X = T::add_mod(X, &temp, q);
}

fn opt_butterfly(){

}


fn inverse_butterfly(X: &mut Scalar, Y: &mut Scalar){

}

#[cfg(test)]
mod tests {
    use super::*;


    fn test_butterfly(arr: [u64;4]) -> [u64; 2] {
      let mut X:Scalar = Scalar::from(arr[0]);
      let mut Y:Scalar = Scalar::from(arr[1]);
      let W:Scalar = Scalar::from(arr[2]);
      let q:Scalar = Scalar::new_modulus(arr[3]);

      butterfly(&mut X, &mut Y, &W, &q);
      [X.into(), Y.into()]
    }

    fn fib(arr: [u8;2]) -> u8{
      arr[0]
    }

    macro_rules! butterfly_tests {
      ($($name:ident: $value:expr,)*) => {
      $(
          #[test]
          fn $name() {
              let (input, expected) = $value;
              assert_eq!(expected, test_butterfly(input));
          }
      )*
    }
  }

  butterfly_tests! {
    butterfly_0: ([0u64, 1u64, 0u64, 100u64], [0u64, 0u64]),
    butterfly_1: ([0u64, 1u64, 0u64, 100u64], [0u64, 0u64]),
    butterfly_2: ([0u64, 1u64, 0u64, 100u64], [0u64, 0u64]),
    butterfly_3: ([0u64, 1u64, 0u64, 100u64], [0u64, 0u64]),
  }
}
