// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#[cfg(test)]
use super::{SuperTrait};

use super::{ArithUtils};

// (X, Y) -> (X+Y, W(X-Y)) mod q
#[allow(non_snake_case)]
pub fn inverse_butterfly<T>(X: &mut T, Y: &mut T, W: &T, q: &T) where T: ArithUtils<T>{
    let temp  = T::sub_mod(X,Y, q);
    *X = T::add_mod(X, Y, q);
    *Y = T::mul_mod(W, &temp, q);
}

// (X, Y) -> (X+WY, X-WY) mod q
#[allow(non_snake_case)]
pub fn butterfly<T>(X: &mut T, Y: &mut T, W: &T, q: &T) where T: ArithUtils<T>{
    let temp  = T::mul_mod(Y, W, q);
    *Y = T::sub_mod(X, &temp, q);
    *X = T::add_mod(X, &temp, q);
}

// (X, Y) -> (X+WY, X-WY)
// 0 <= X, Y < 4q => (0 <= X', Y' < 4q)
#[allow(non_snake_case)]
#[cfg(test)]
pub fn lazy_butterfly<T>(X: &mut T, Y: &mut T, W: u64, Wprime: u64, q: &T, twoq: u64) where T: SuperTrait<T>{
    let mut xx = X.rep();
    if xx > twoq{
        xx -= twoq;
    }
    let _qq = super::util::mul_high_word(Wprime, Y.rep());
    let quo = W.wrapping_mul(Y.rep()) - _qq.wrapping_mul(q.rep());
    // X += quo;
    *X = T::from(xx + quo);
    // Y = (x + 2q - quo);
    *Y = T::from(xx + twoq - quo);
}

#[allow(clippy::many_single_char_names)]
pub fn lazy_butterfly_u64(mut x: u64, y:u64, w: u64, wprime: u64, q: u64, twoq: u64) -> (u64, u64){
    // let twoq = 0;
    if x > twoq{
        x -= twoq;
    }
    let _qq = super::util::mul_high_word(wprime, y);
    let wy = w.wrapping_mul(y);
    let qqq = _qq.wrapping_mul(q);
    let quo;
    if wy >= qqq {
        quo = wy - qqq;
    }
    else{
        quo = u64::MAX - qqq + wy + 1;
    }
    (x + quo, x + twoq - quo)
}

#[allow(clippy::many_single_char_names)]
pub fn lazy_inverse_butterfly_u64(x: u64, y:u64, w: u64, wprime: u64, q: u64, twoq: u64) -> (u64, u64){
    let mut xx = x+y;

    if xx > twoq {
        xx -= twoq;
    }
    let t = twoq - y + x;
    let quo = super::util::mul_high_word(wprime, t);
    let wt = w.wrapping_mul(t);
    let qquo = quo.wrapping_mul(q);
    let yy;
    if wt >= qquo {
        yy = wt - qquo;
    }
    else{
        yy = u64::MAX - qquo + wt + 1;
    }
    (xx, yy)
}

// (X,Y) -> (X+Y, W(X-Y)) mod q
// 0 <= X, Y < 2q  ==> 0 <= X', Y' < 2q
#[allow(non_snake_case)]
#[cfg(test)]
pub(crate) fn lazy_inverse_butterfly<T>(X: &mut T, Y: &mut T, W: u64, Wprime: u64, q: &T) where T: SuperTrait<T>{
    let mut xx = X.rep() + Y.rep();

    let twoq = 2*q.rep();
    if xx > twoq {
        xx -= twoq;
    }
    let t = twoq - Y.rep() + X.rep();
    let quo = super::util::mul_high_word(Wprime, t);
    let yy = W.wrapping_mul(t) - quo.wrapping_mul(q.rep());
    *X = T::from(xx);
    *Y = T::from(yy);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::scalar::Scalar;

    fn butterfly_for_test(arr: [u64;4]) -> [u64; 2] {
      let mut x:Scalar = Scalar::from(arr[0]);
      let mut y:Scalar = Scalar::from(arr[1]);
      let w:Scalar = Scalar::from(arr[2]);
      let q:Scalar = Scalar::new_modulus(arr[3]);

      butterfly(&mut x, &mut y, &w, &q);
      [x.into(), y.into()]
    }

    fn inverse_butterfly_for_test(arr: [u64;4]) -> [u64; 2] {
      let mut x:Scalar = Scalar::from(arr[0]);
      let mut y:Scalar = Scalar::from(arr[1]);
      let w:Scalar = Scalar::from(arr[2]);
      let q:Scalar = Scalar::new_modulus(arr[3]);

      inverse_butterfly(&mut x, &mut y, &w, &q);
      [x.into(), y.into()]
    }

    fn lazy_butterfly_for_test(arr: [u64;4]) -> [u64; 2] {
      let mut x:Scalar = Scalar::from(arr[0]);
      let mut y:Scalar = Scalar::from(arr[1]);
      let w = arr[2];
      let q:Scalar = Scalar::new_modulus(arr[3]);
      //  W′ = ⌊W β/p⌋, 0 < W′ < β
      let wprime: u64 = super::super::util::compute_harvey_ratio(w, q.rep());
      let twoq = q.rep() << 1;

      lazy_butterfly(&mut x, &mut y, w, wprime, &q, twoq);
      [x.into(), y.into()]
    }

    fn lazy_inverse_butterfly_for_test(arr: [u64;4]) -> [u64; 2] {
      let mut x:Scalar = Scalar::from(arr[0]);
      let mut y:Scalar = Scalar::from(arr[1]);
      let w = arr[2];
      let q:Scalar = Scalar::new_modulus(arr[3]);
      //  W′ = ⌊W β/p⌋, 0 < W′ < β
      let wprime: u64 = super::super::util::compute_harvey_ratio(w, q.rep());

      lazy_inverse_butterfly(&mut x, &mut y, w, wprime, &q);
      [x.into(), y.into()]
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
    lazy_butterfly_0: [0u64, 1u64, 0u64, 100u64],
    lazy_butterfly_1: [1u64, 1u64, 1u64, 100u64],
    lazy_butterfly_2: [50u64, 50u64, 1u64, 100u64],
    lazy_butterfly_3: [1u64, 1u64, 50u64, 100u64],
  }

  lazy_inverse_butterfly_tests! {
    lazy_inverse_butterfly_0: [0u64, 1u64, 0u64, 100u64],
    lazy_inverse_butterfly_1: [1u64, 1u64, 1u64, 100u64],
    lazy_inverse_butterfly_2: [50u64, 50u64, 1u64, 100u64],
    lazy_inverse_butterfly_3: [1u64, 1u64, 50u64, 100u64],
  }
}
