// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

/// computes floor(a*b/pow(2,64))
pub fn mul_high_word(a: u64, b:u64) -> u64{
    ((a as u128 * b as u128) >> 64) as u64
}

/// computes floor(w*pow(2,64)/q)
pub fn compute_harvey_ratio(w: u64, q: u64) -> u64{
    (((w as u128) << 64 )/ q as u128) as u64
}

pub fn mul_low_word(a: u64, b: u64) -> u64 {
    let res = (a as u128) * (b as u128);
    (res >> 64) as u64
}

#[cfg(test)]
mod tests {
  use super::*;


  #[test]
  fn test_mul_high_word(){
    assert_eq!(mul_high_word(1,1), 0);
    assert_eq!(mul_high_word(1u64 << 63,0), 0);
    assert_eq!(mul_high_word(1u64 << 63,2), 1);
    assert_eq!(mul_high_word(1u64 << 63,1u64 << 63), 1u64 << 62);
  }

  #[test]
  fn test_compute_harvey_ratio(){
    assert_eq!(compute_harvey_ratio(0,100), 0);
    assert_eq!(compute_harvey_ratio(1,100), 184467440737095516);
  }
}
