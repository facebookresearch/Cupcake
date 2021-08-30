use crate::integer_arith::butterfly::{lazy_butterfly_u64, lazy_inverse_butterfly_u64}; 

pub fn lazy_ntt_u64(vec: &mut [u64], roots: &[u64], scaled_roots: &[u64], q: u64){
    let n = vec.len(); 
    let twoq = q << 1; 

    let mut t = n;
    let mut m = 1;
    while m < n {
        t >>= 1;
        for i in 0..m {
            let j1 = 2 * i * t;
            let j2 = j1 + t - 1;
            let phi = roots[m + i];
            for j in j1..j2 + 1 {
                let new = lazy_butterfly_u64(vec[j], vec[j+t], phi, scaled_roots[m+i], q, twoq);
                vec[j] = new.0; 
                vec[j+t] = new.1; 
            }
        }
        m <<= 1;
    }
}

pub fn lazy_inverse_ntt_u64(vec: &mut [u64], invroots: &[u64], scaled_invroots: &[u64], q: u64){
    let twoq = q << 1; 

    let mut t = 1;
    let mut m = vec.len();
    while m > 1 {
        let mut j1 = 0;
        let h = m >> 1;
        for i in 0..h {
            let j2 = j1 + t - 1;
            for j in j1..j2 + 1 {
                // inverse butterfly
                let new = lazy_inverse_butterfly_u64(vec[j], vec[j+t], invroots[h+i], scaled_invroots[h+i], q, twoq);
                vec[j] = new.0; 
                vec[j+t] = new.1; 
            }
            j1 += 2 * t;
        }
        t <<= 1;
        m >>= 1;
    }
}