use crate::integer_arith::butterfly::{lazy_butterfly_u64, lazy_inverse_butterfly_u64}; 

pub fn lazy_ntt_u64(v: &mut Vec<u64>, roots: &Vec<u64>, scaled_roots: &Vec<u64>, q: u64){
    let n = v.len(); 
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
                let new = lazy_butterfly_u64(v[j], v[j+t], phi, scaled_roots[m+i], q, twoq);
                v[j] = new.0; 
                v[j+t] = new.1; 
            }
        }
        m <<= 1;
    }
}

pub fn lazy_inverse_ntt_u64(v: &mut Vec<u64>, invroots: &Vec<u64>, scaled_invroots: &Vec<u64>, q: u64){
    let n = v.len(); 
    let twoq = q << 1; 

    let mut t = 1;
    let mut m = n;
    while m > 1 {
        let mut j1 = 0;
        let h = m >> 1;
        for i in 0..h {
            let j2 = j1 + t - 1;
            for j in j1..j2 + 1 {
                // inverse butterfly
                let new = lazy_inverse_butterfly_u64(v[j], v[j+t], invroots[h+i], scaled_invroots[h+i], q, twoq);
                v[j] = new.0; 
                v[j+t] = new.1; 
            }
            j1 += 2 * t;
        }
        t <<= 1;
        m >>= 1;
    }
}