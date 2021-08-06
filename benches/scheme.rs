// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#[macro_use]
extern crate bencher;
pub use std::sync::Arc;
use bencher::Bencher;
use cupcake::traits::*;
use cupcake::integer_arith::scalar::Scalar;
use cupcake::integer_arith::ArithUtils;


fn encrypt_sk(bench: &mut Bencher) {
    let fv = cupcake::default();

    let sk = fv.generate_key();

    let mut v = vec![0; fv.n];
    for i in 0..fv.n {
        v[i] = i as u8;
    }
    bench.iter(|| {
        let _ = fv.encrypt_sk(&v, &sk);
    })
}

fn decryption(bench: &mut Bencher) {
    let fv = cupcake::default();

    let sk = fv.generate_key();
    let mut v = vec![0; fv.n];
    for i in 0..fv.n {
        v[i] = i as u8;
    }
    let ct = fv.encrypt_sk(&v, &sk);
    bench.iter(|| {
        let _: Vec<u8> = fv.decrypt(&ct, &sk);
    })
}

fn encrypt_pk(bench: &mut Bencher) {
    let fv = cupcake::default();

    let (pk, _sk) = fv.generate_keypair();
    let mut v = vec![0; fv.n];
    for i in 0..fv.n {
        v[i] = i as u8;
    }
    bench.iter(|| {
        let _ = fv.encrypt(&v, &pk);
    })
}

fn encrypt_zero_pk(bench: &mut Bencher) {
    let fv = cupcake::default();

    let (pk, _sk) = fv.generate_keypair();
    let mut v = vec![0; fv.n];
    for i in 0..fv.n {
        v[i] = i as u8;
    }
    bench.iter(|| {
        let _ = fv.encrypt_zero(&pk);
    })
}

fn homomorphic_addition(bench: &mut Bencher) {
    let fv = cupcake::default();

    let sk = fv.generate_key();

    let mut v = vec![0; fv.n];
    for i in 0..fv.n {
        v[i] = i as u8;
    }
    let mut ct1 = fv.encrypt_sk(&v, &sk);
    let ct2 = fv.encrypt_sk(&v, &sk);
    bench.iter(|| {
        fv.add_inplace(&mut ct1, &ct2);
    })
}

fn rerandomize(bench: &mut Bencher) {
    let fv = cupcake::default();

    let (pk, _) = fv.generate_keypair();
    let mut v = vec![0; fv.n];
    for i in 0..fv.n {
        v[i] = i as u8;
    }
    let mut ct = fv.encrypt(&v, &pk);

    bench.iter(|| {
        fv.rerandomize(&mut ct, &pk);
    })
}

benchmark_group!(
    scheme,
    encrypt_sk,
    encrypt_pk,
    encrypt_zero_pk,
    decryption,
    homomorphic_addition,
    rerandomize,
);

benchmark_main!(scheme);
