// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
pub(crate) fn reverse_bits_perm<T>(input: &mut Vec<T>) {
    let n = input.len();
    if !n.is_power_of_two() {
        panic!("n must be a power of 2");
    }
    for i in 0..n {
        let j = bit_reverse(i, n);
        if j > i {
            input.swap(i, j);
        }
    }
}

pub(crate) fn bit_reverse(i: usize, n: usize) -> usize {
    let mut mask = 1;
    let mut j = 0;
    while mask < n {
        let t = (i & mask) != 0;
        j += t as usize;
        mask <<= 1;
        if mask < n {
            j <<= 1;
        }
    }
    j
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 4), 0);
        assert_eq!(bit_reverse(1, 4), 2);
        assert_eq!(bit_reverse(2, 4), 1);
        assert_eq!(bit_reverse(3, 4), 3);

        assert_eq!(bit_reverse(0, 8), 0);
        assert_eq!(bit_reverse(1, 8), 4);
        assert_eq!(bit_reverse(2, 8), 2);
        assert_eq!(bit_reverse(3, 8), 6);
        assert_eq!(bit_reverse(4, 8), 1);
        assert_eq!(bit_reverse(5, 8), 5);
        assert_eq!(bit_reverse(6, 8), 3);
        assert_eq!(bit_reverse(7, 8), 7);
    }
}
