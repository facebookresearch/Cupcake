// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use cupcake::traits::*;
use cupcake::DefaultShemeType;
use cupcake::FVCiphertext;
use cupcake::FVPlaintext;
use cupcake::integer_arith::scalar::Scalar;
use cupcake::SecretKey;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;

use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, Error, Formatter};

#[derive(Hash, PartialEq, Eq, Clone, Debug, Serialize, Deserialize)]
pub struct ByteBuffer {
    pub buffer: Vec<u8>,
}

impl ByteBuffer {
    pub fn from_slice(v: &[u8]) -> ByteBuffer {
        ByteBuffer { buffer: v.to_vec() }
    }
}

impl Display for ByteBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        //todo : fix me with proper formatting
        for b in self.buffer.iter() {
            write!(f, "{:X}", b)?;
        }
        Ok(())
    }
}

fn smartprint<T: std::fmt::Debug>(v: &Vec<T>) {
    println!("[{:?}, {:?}, ..., {:?}]", v[0], v[1], v[v.len() - 1]);
}

fn smartprint2<T: std::fmt::Debug>(v: &Vec< Vec<T>>) {
    smartprint(&v[0]);
    smartprint(&v[1]);
    smartprint(&v[v.len() -1]);
}


// a struct for Parallel Cupcake operations

    // usage for main method PS3I outline:
    //      cargo run --example parallel
    //      cargo run --release --example parallel
    // usage for CupcakeParallel tests
    //      cargo test --examples parallel
    //      cargo test --release --examples parallel

pub struct CupcakeParallel {
    pub scheme: DefaultShemeType,
    pub pk: FVCiphertext<Scalar>,
    sk: SecretKey<Scalar>
}

impl CupcakeParallel {
    pub fn new() -> CupcakeParallel {
        let scheme_ = cupcake::default();
        let (pk_, sk_) = scheme_.generate_keypair();
        CupcakeParallel {
            scheme: scheme_,
            pk: pk_,
            sk: sk_
        }
    }

    pub fn parallel_encrypt_serialize(&self, plaintext:&Vec< Vec<u8> >) -> Vec<ByteBuffer> {
        plaintext.into_par_iter().map(|item| {
            let t = &self.scheme.encrypt(&*item, &self.pk);
            ByteBuffer {
                buffer: t.to_bytes()
            }
        })
        .collect::<Vec<ByteBuffer>>()
    }

    pub fn parallel_deserialize_decrypt(&self, payload: &Vec<ByteBuffer>) -> Vec<Vec<u8>> {
        let scheme_ = &self.scheme;
        let sk_ = &self.sk;
        payload.into_par_iter().map(|item| {
                let pt = scheme_.decrypt(&scheme_.from_bytes(&(item.buffer)), &sk_);
                return pt;
            })
            .collect::<Vec<Vec<u8>>>()
    }

    // also deserializes and serializes at both ends.
    pub fn parallel_plaintext_ciphertext_subtract_rerandomize(&self, ctvec: &Vec<ByteBuffer>, ptvec: &Vec<Vec<u8>>,
    ) -> Vec<ByteBuffer> {
        let pk_ = &self.pk;
        let it_ct = ctvec.into_par_iter();
        let it_pt = ptvec.into_par_iter();
        let scheme_ = &self.scheme;
        it_ct
            .zip_eq(it_pt)
            .map(|(ct_bytes, pt)| {
                let mut ct:FVCiphertext<Scalar> = scheme_.from_bytes(&ct_bytes.buffer);
                &self.scheme.sub_plain_inplace( &mut ct, pt); // ct := ct - pt
                &self.scheme.rerandomize(&mut ct, &pk_);
                ByteBuffer {
                    buffer: ct.to_bytes()
                }
            })
            .collect::<Vec<ByteBuffer>>()
    }

        // also deserializes and serializes at both ends.
    pub fn parallel_plaintext_ciphertext_add_rerandomize(&self, ctvec: &Vec<ByteBuffer>, ptvec: &Vec<Vec<u8>>,
    ) -> Vec<ByteBuffer> {
        let pk_ = &self.pk;
        let it_ct = ctvec.into_par_iter();
        let it_pt = ptvec.into_par_iter();
        let scheme_ = &self.scheme;
        it_ct
            .zip_eq(it_pt)
            .map(|(ct_bytes, pt)| {
                let mut ct:FVCiphertext<Scalar> = scheme_.from_bytes(&ct_bytes.buffer);
                &self.scheme.add_plain_inplace( &mut ct, pt); // adds pt into ct
                &self.scheme.rerandomize(&mut ct, &pk_);
                ByteBuffer {
                    buffer: ct.to_bytes()
                }
            })
            .collect::<Vec<ByteBuffer>>()
    }

    pub fn parallel_plaintext_ciphertext_subtract(&self, ctvec: &Vec<ByteBuffer>, ptvec: &Vec<Vec<u8>>,
    ) -> Vec<ByteBuffer> {
        let it_ct = ctvec.into_par_iter();
        let it_pt = ptvec.into_par_iter();
        let scheme_ = &self.scheme;
        it_ct
            .zip_eq(it_pt)
            .map(|(ct_bytes, pt)| {
                let mut ct:FVCiphertext<Scalar> = scheme_.from_bytes(&ct_bytes.buffer);
                &self.scheme.sub_plain_inplace( &mut ct, pt); // subtracts pt into ct
                ByteBuffer {
                    buffer: ct.to_bytes()
                }
            })
            .collect::<Vec<ByteBuffer>>()
    }

    // would it be more efficient to pass by mutable reference and have no return??
    pub fn parallel_plaintext_ciphertext_add(&self, ctvec: &Vec<ByteBuffer>, ptvec: &Vec<Vec<u8>>,
    ) -> Vec<ByteBuffer> {
        let it_ct = ctvec.into_par_iter();
        let it_pt = ptvec.into_par_iter();
        let scheme_ = &self.scheme;
        it_ct
            .zip_eq(it_pt)
            .map(|(ct_bytes, pt)| {
                let mut ct:FVCiphertext<Scalar> = scheme_.from_bytes(&ct_bytes.buffer);
                &self.scheme.add_plain_inplace( &mut ct, pt); // adds pt into ct
                ByteBuffer {
                    buffer: ct.to_bytes()
                }
            })
            .collect::<Vec<ByteBuffer>>()
    }

    pub fn simple_encrypt(&self, plaintext: &FVPlaintext<u8>) -> FVCiphertext<Scalar> {
        let scheme_ = &self.scheme;
        let pk_ = &self.pk;
        let ct = scheme_.encrypt(plaintext, &pk_);
        return ct;
    }

    pub fn simple_plaintext_ciphertext_add(&self, ct:&mut FVCiphertext<Scalar>, pt: &FVPlaintext<u8>  ){
        let scheme_ = &self.scheme;
        scheme_.add_plain_inplace( ct, pt);
    }
}



fn main() {
    // Oultine of PS3I HE operations using cupcake

    //////////////  SETUP ///////////////////

    // company and partner generate HE key pairs
    let parfv_company = CupcakeParallel::new();
    let parfv_partner = CupcakeParallel::new();

    // TODO: what we would really do is serialize the schemes and public keys and
    //       exchange; then have the recieving party create their own CupcakeParallel
    //       object initialized with the other party's scheme and pk and with a zero sk

    //////////////  Step 1 Company ///////////////////
    let n = parfv_partner.scheme.n;
    let  v_c:Vec<Vec<u8>> = vec![ vec![10;n ],
                                  vec![20;n ],
                                  vec![30;n ]];

    let ctv_c = parfv_company.parallel_encrypt_serialize(&v_c);

    println!("v_c: ");
    smartprint2(&v_c);

    //////////////  Step 1 Partner ///////////////////
    let  v_p:Vec<Vec<u8>>  = vec![ vec![40;n ],
                                   vec![50;n ],
                                   vec![60;n ]];

    let ctv_p = parfv_partner.parallel_encrypt_serialize(&v_p);

    println!("v_p: ");
    smartprint2(&v_p);

    let  r_c:Vec<Vec<u8>>  = vec![ vec![1;n ],
                                   vec![2;n ],
                                   vec![3;n ]];

    let ctv_c_double_prime = parfv_company.parallel_plaintext_ciphertext_subtract_rerandomize(&ctv_c, &r_c);

    println!("r_c: ");
    smartprint2(&r_c);

    // //////////////  Step 2 Company ///////////////////
    let  r_p:Vec<Vec<u8>>  = vec![ vec![4;n ],
                                   vec![5;n ],
                                   vec![6;n ]];

    let ctv_p_double_prime = parfv_partner.parallel_plaintext_ciphertext_subtract_rerandomize(&ctv_p, &r_p);

    println!("r_p: ");
    smartprint2(&r_p);

    let v_c_double_prime = parfv_company.parallel_deserialize_decrypt(&ctv_c_double_prime);

    println!("decrypted v_c_double_prime = v_c - r_c: ");
    smartprint2(&v_c_double_prime);

    // //////////////  Step 2 Partner ///////////////////
    let v_p_double_prime = parfv_partner.parallel_deserialize_decrypt(&ctv_p_double_prime);

    println!("decrypted v_p_double_prime = v_p - r_p: ");
    smartprint2(&v_p_double_prime);


}


#[cfg(test)]
mod fv_scalar_tests {
    use super::*;
    #[test]
    fn test_parallel_encrypt_decrypt() {
        let parallelfv = CupcakeParallel::new();

        let  v = vec![ vec![1;parallelfv.scheme.n ],
                       vec![2;parallelfv.scheme.n ],
                       vec![3;parallelfv.scheme.n ]];

        let ctbytes = parallelfv.parallel_encrypt_serialize(&v);
        let pt_actual = parallelfv.parallel_deserialize_decrypt(&ctbytes);

        assert_eq!(v, pt_actual);
    }

    #[test]
    fn test_parallel_add() {
        let parallelfv = CupcakeParallel::new();

        let  v = vec![ vec![1;parallelfv.scheme.n ],
                       vec![2;parallelfv.scheme.n ],
                       vec![200;parallelfv.scheme.n ]];

        let  w = vec![ vec![10;parallelfv.scheme.n ],
                       vec![20;parallelfv.scheme.n ],
                       vec![70;parallelfv.scheme.n ]];

        let ctvbytes = parallelfv.parallel_encrypt_serialize(&v);
        let ctv_wbytes = parallelfv.parallel_plaintext_ciphertext_add(&ctvbytes, &w);

        let pt_actual = parallelfv.parallel_deserialize_decrypt(&ctv_wbytes);

        let mut vplusw = vec![ vec![0;parallelfv.scheme.n ],
                         vec![0;parallelfv.scheme.n ],
                         vec![0;parallelfv.scheme.n ]];
        for i in 0..v.len(){
            for j in 0..v[i].len(){
                vplusw[i][j]=u8::wrapping_add(v[i][j],w[i][j]);
            }
        }
        assert_eq!(vplusw, pt_actual);
    }

    #[test]
    fn test_parallel_plaintext_ciphertext_subtract_rerandomize() {
        let parallelfv = CupcakeParallel::new();

        let  v = vec![ vec![1;parallelfv.scheme.n ],
                       vec![2;parallelfv.scheme.n ],
                       vec![200;parallelfv.scheme.n ]];

        let  w = vec![ vec![10;parallelfv.scheme.n ],
                       vec![20;parallelfv.scheme.n ],
                       vec![70;parallelfv.scheme.n ]];

        let ctvbytes = parallelfv.parallel_encrypt_serialize(&v);
        let ctv_wbytes = parallelfv.parallel_plaintext_ciphertext_subtract_rerandomize(&ctvbytes, &w);

        let pt_actual = parallelfv.parallel_deserialize_decrypt(&ctv_wbytes);

        let mut vminuw = vec![ vec![0;parallelfv.scheme.n ],
                               vec![0;parallelfv.scheme.n ],
                               vec![0;parallelfv.scheme.n ]];
        for i in 0..v.len(){
            for j in 0..v[i].len(){
                vminuw[i][j]=u8::wrapping_sub(v[i][j],w[i][j]);
            }
        }
        assert_eq!(vminuw, pt_actual);
    }

    #[test]
    fn test_parallel_plaintext_ciphertext_add_rerandomize() {
        let parallelfv = CupcakeParallel::new();

        let  v = vec![ vec![1;parallelfv.scheme.n ],
                       vec![2;parallelfv.scheme.n ],
                       vec![200;parallelfv.scheme.n ]];

        let  w = vec![ vec![10;parallelfv.scheme.n ],
                       vec![20;parallelfv.scheme.n ],
                       vec![70;parallelfv.scheme.n ]];

        let ctvbytes = parallelfv.parallel_encrypt_serialize(&v);
        let ctv_wbytes = parallelfv.parallel_plaintext_ciphertext_add_rerandomize(&ctvbytes, &w);

        let pt_actual = parallelfv.parallel_deserialize_decrypt(&ctv_wbytes);

        let mut vplusw = vec![ vec![0;parallelfv.scheme.n ],
                               vec![0;parallelfv.scheme.n ],
                               vec![0;parallelfv.scheme.n ]];
        for i in 0..v.len(){
            for j in 0..v[i].len(){
                vplusw[i][j]=u8::wrapping_add(v[i][j],w[i][j]);
            }
        }
        assert_eq!(vplusw, pt_actual);
    }
}
