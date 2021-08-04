// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::scalar::Scalar;
use super::{SuperTrait, ArithUtils};

// (X, Y) -> (X+Y, W(X-Y)) mod q 
fn inverse_butterfly<T>(X: &mut T, Y: &mut T, W: &T, q: &T) where T: ArithUtils<T>{
    let temp  = T::sub_mod(X,Y, q);
    *X = T::add_mod(X, &Y, q);
    *Y = T::mul_mod(W, &temp, q);
}

// (X, Y) -> (X+WY, X-WY) mod q 
pub(crate) fn butterfly<T>(X: &mut T, Y: &mut T, W: &T, q: &T) where T: ArithUtils<T>{
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

// (X,Y) -> (X+Y, W(X-Y)) mod q
// 0 <= X, Y < 2q  ==> 0 <= X', Y' < 2q 
fn lazy_inverse_butterfly<T>(X: &mut T, Y: &mut T, W: u64, Wprime: u64, q: &T) where T: SuperTrait<T>{
    let mut xx = X.rep() + Y.rep(); 
    	
    let twoq = 2*q.rep();
    if xx > twoq {
        xx -= twoq; 
    }
    let t = twoq - Y.rep() + X.rep(); 
    let quo = super::util::mul_high_word(Wprime, t); 
    println!("quo = {}", quo);
    println!("wprime = {}", Wprime);
    let yy: u64 = super::util::mul_low_word(W,t) - super::util::mul_low_word(quo,q.rep()); 
    *X = T::from(xx); 
    *Y = T::from(yy);
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

    fn inverse_butterfly_for_test(arr: [u64;4]) -> [u64; 2] {
      let mut X:Scalar = Scalar::from(arr[0]);
      let mut Y:Scalar = Scalar::from(arr[1]);
      let W:Scalar = Scalar::from(arr[2]);
      let q:Scalar = Scalar::new_modulus(arr[3]);

      inverse_butterfly(&mut X, &mut Y, &W, &q);
      [X.into(), Y.into()]
    }


    fn lazy_butterfly_for_test(arr: [u64;4]) -> [u64; 2] {
      let mut X:Scalar = Scalar::from(arr[0]);
      let mut Y:Scalar = Scalar::from(arr[1]);
      let W = arr[2];
      let q:Scalar = Scalar::new_modulus(arr[3]);
      //  W′ = ⌊W β/p⌋, 0 < W′ < β
      let Wprime: u64 = super::super::util::compute_harvey_ratio(W, q.rep());

      lazy_butterfly(&mut X, &mut Y, W, Wprime, &q);
      [X.into(), Y.into()]
    }

    fn lazy_inverse_butterfly_for_test(arr: [u64;4]) -> [u64; 2] {
      let mut X:Scalar = Scalar::from(arr[0]);
      let mut Y:Scalar = Scalar::from(arr[1]);
      let W = arr[2];
      let q:Scalar = Scalar::new_modulus(arr[3]);
      //  W′ = ⌊W β/p⌋, 0 < W′ < β
      let Wprime: u64 = super::super::util::compute_harvey_ratio(W, q.rep());

      lazy_inverse_butterfly(&mut X, &mut Y, W, Wprime, &q);
      [X.into(), Y.into()]
    }

    macro_rules! lazy_butterfly_tests {
      ($($name:ident: $value:expr,)*) => {
      $(
          #[test]
          fn $name() {
              let input = $value;
              let butterfly_out = butterfly_for_test(input); 
              let output = lazy_butterfly_for_test(input);
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

    macro_rules! lazy_inverse_butterfly_tests {
      ($($name:ident: $value:expr,)*) => {
      $(
          #[test]
          fn $name() {
              let input = $value;
              let butterfly_out = inverse_butterfly_for_test(input); 
              let output = lazy_inverse_butterfly_for_test(input);
              println!("{:?}", butterfly_out); 
              println!("{:?}", output); 
              assert!(output[0] < 2*input[3]); 
              assert!(output[1] < 2*input[3]); 
              assert_eq!((output[1] - butterfly_out[1]) % input[3], 0); 
              assert_eq!((output[0] - butterfly_out[0]) % input[3], 0); 
          }
        )*
      }
    }

    macro_rules! inverse_butterfly_tests {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (input, expected) = $value;
                assert_eq!(expected, inverse_butterfly_for_test(input));
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

  inverse_butterfly_tests! {
    inverse_butterfly_0: ([0u64, 1u64, 0u64, 100u64], [1u64, 0u64]),
    inverse_butterfly_1: ([1u64, 1u64, 1u64, 100u64], [2u64, 0u64]),
    inverse_butterfly_2: ([50u64, 50u64, 1u64, 100u64], [0u64, 0u64]),
    inverse_butterfly_3: ([2u64, 1u64, 50u64, 100u64], [3u64, 50u64]),
  }

  lazy_butterfly_tests! {
    lazy_butterfly_0: ([0u64, 1u64, 0u64, 100u64]),
    lazy_butterfly_1: ([1u64, 1u64, 1u64, 100u64]),
    lazy_butterfly_2: ([50u64, 50u64, 1u64, 100u64]),
    lazy_butterfly_3: ([1u64, 1u64, 50u64, 100u64]),
  }

  lazy_inverse_butterfly_tests! {
    lazy_inverse_butterfly_0: ([0u64, 1u64, 0u64, 100u64]),
    lazy_inverse_butterfly_1: ([1u64, 1u64, 1u64, 100u64]),
    lazy_inverse_butterfly_2: ([50u64, 50u64, 1u64, 100u64]),
    lazy_inverse_butterfly_3: ([1u64, 1u64, 50u64, 100u64]),
  }
}
