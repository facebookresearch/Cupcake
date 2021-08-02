
/// computes floor(a*b/pow(2,64))
pub fn mul_high_word(a: u64, b:u64) -> u64{
  return ((a as u128 * b as u128) >> 64) as u64;
}

/// computes floor(w*pow(2,64)/q)
pub fn compute_harvey_ratio(w: u64, q: u64) -> u64{
  return ((w as u128) << 64 / q) as u64; 
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
    assert_eq!(compute_harvey_ratio(0,1), 0);
  }
}
