// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::scalar::Scalar;
use super::{SuperTrait, ArithUtils};

// (X, Y) -> (X+WY, X-WY) mod q 
fn butterfly<T>(X: &mut T, Y: &mut T, W: &T, q: &T) where T: ArithUtils<T>{
  let temp  = T::mul_mod(Y, W, q);
  *Y = T::sub_mod(X, &temp, q);
  *X = T::add_mod(X, &temp, q);
}

// (X, Y) -> (X+WY, X-WY)
// 0 <= X, Y < 4q => (0 <= X', Y' < 4q)
fn lazy_butterfly<T>(X: &mut T, Y: &mut T, W: u64, Wprime: u64, q: &T) where T: SuperTrait<T>{
  let twoq = 2*q.rep();

  if X.rep() > twoq{
    X.sub_u64(twoq);
  }
  let xx = X.rep();
  let _qq = super::util::mul_high_word(Wprime, Y.rep());
  let quo = W * Y.rep() - _qq * q.rep();
  println!("quo = {}", quo);
  // X += quo;
  X.add_u64(quo);
  // Y += (2q - quo);
  println!("Y = {}", Y.rep());
  *Y = T::from(xx + twoq - quo); 
}


fn opt_butterfly(){
  todo!()
}


fn inverse_butterfly(X: &mut Scalar, Y: &mut Scalar){
  todo!()
}




#[cfg(test)]
mod tests {
    use super::*;


    fn butterfly_for_test(arr: [u64;4]) -> [u64; 2] {
      let mut X:Scalar = Scalar::from(arr[0]);
      let mut Y:Scalar = Scalar::from(arr[1]);
      let W:Scalar = Scalar::from(arr[2]);
      let q:Scalar = Scalar::new_modulus(arr[3]);

      butterfly(&mut X, &mut Y, &W, &q);
      [X.into(), Y.into()]
    }


    fn test_lazy_butterfly(arr: [u64;4]) -> [u64; 2] {
      let mut X:Scalar = Scalar::from(arr[0]);
      let mut Y:Scalar = Scalar::from(arr[1]);
      let W = arr[2];
      let q:Scalar = Scalar::new_modulus(arr[3]);
      //  W′ = ⌊W β/p⌋, 0 < W′ < β
      let Wprime: u64 = super::super::util::compute_harvey_ratio(W, q.rep());

      lazy_butterfly(&mut X, &mut Y, W, Wprime, &q);
      [X.into(), Y.into()]
    }

    macro_rules! lazy_butterfly_tests {
      ($($name:ident: $value:expr,)*) => {
      $(
          #[test]
          fn $name() {
              let input = $value;
              let butterfly_out = butterfly_for_test(input); 
              let output = test_lazy_butterfly(input);
              println!("{:?}", butterfly_out); 
              println!("{:?}", output); 
              assert!(output[0] < 4*input[3]); 
              assert!(output[1] < 4*input[3]); 
              assert_eq!((output[1] - butterfly_out[1]) % input[3], 0); 
              assert_eq!((output[0] - butterfly_out[0]) % input[3], 0); 
          }
        )*
      }
    }

    macro_rules! butterfly_tests {
      ($($name:ident: $value:expr,)*) => {
      $(
          #[test]
          fn $name() {
              let (input, expected) = $value;
              assert_eq!(expected, butterfly_for_test(input));
          }
        )*
      }
    }

  butterfly_tests! {
    butterfly_0: ([0u64, 1u64, 0u64, 100u64], [0u64, 0u64]),
    butterfly_1: ([1u64, 1u64, 1u64, 100u64], [2u64, 0u64]),
    butterfly_2: ([50u64, 50u64, 1u64, 100u64], [0u64, 0u64]),
    butterfly_3: ([1u64, 1u64, 50u64, 100u64], [51u64, 51u64]),
  }

  lazy_butterfly_tests! {
    lazy_butterfly_0: ([0u64, 1u64, 0u64, 100u64]),
    lazy_butterfly_1: ([1u64, 1u64, 1u64, 100u64]),
    lazy_butterfly_2: ([50u64, 50u64, 1u64, 100u64]),
    lazy_butterfly_3: ([1u64, 1u64, 50u64, 100u64]),
  }
}
