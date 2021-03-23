// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#[macro_use]
extern crate bencher;

use bencher::Bencher;
use cupcake::traits::*;
// use cupcake::rqpoly::{RqPoly, RqPolyContext};
// use rand::rngs::StdRng;
// use rand::FromEntropy;

// fn mod_mul(bench: &mut Bencher) {
//     let fv = FV::<BigInt>::default_2048();
//     let a = BigInt::sample_below(&fv.q);
//     let b = BigInt::sample_below(&fv.q);
//     bench.iter(|| {
//         let _ = BigInt::mod_mul(&a, &b, &fv.q);
//     })
// }

// fn scalar_ntt(bench: &mut Bencher) {
//     let q = Scalar::new_modulus(18014398492704769u64);
//     let context = Arc::new(RqPolyContext::new(2048, &q));
//     let mut testpoly = cupcake::randutils::sample_uniform_poly(context.clone());

//     bench.iter(|| {
//         testpoly.is_ntt_form = false;
//         let _ = testpoly.forward_transform();
//     })
// }

// fn scalar_intt(bench: &mut Bencher) {
//     let q = Scalar::new_modulus(18014398492704769u64);
//     let context = Arc::new(RqPolyContext::new(2048, &q));

//     let mut testpoly = cupcake::randutils::sample_uniform_poly(context.clone());

//     bench.iter(|| {
//         testpoly.is_ntt_form = true;
//         let _ = testpoly.inverse_transform();
//     })
// }

// fn mod_mul_fast(bench: &mut Bencher){
//     let q = Scalar::new_modulus(18014398492704769);

//     let ratio = (17592185012223u64, 1024u64);
//     let mut a = Scalar::sample_blw(&q);
//     let mut b = Scalar::sample_blw(&q);
//     a = Scalar::modulus(&a, &q);
//     b = Scalar::modulus(&b, &q);

//     bench.iter(|| {
//         let _ = Scalar::barret_multiply(&a, &b, ratio, q.rep);
//     })
// }

// fn mod_mul_fast_wrap(bench: &mut Bencher) {
//     let q = Scalar::from_u64_raw(18014398492704769);

//     let ratio = (17592185012223u64, 1024u64);
//     let mut a = Scalar::sample_blw(&q);
//     let mut b = Scalar::sample_blw(&q);
//     a = Scalar::modulus(&a, &q);
//     b = Scalar::modulus(&b, &q);

//     bench.iter(|| {
//         let _ = Scalar::mul_mod(&a, &b, &q);
//     })
// }

// fn ntt_multiply(bench: &mut Bencher) {
//     let fv = FV::<BigInt>::default_2048();
//     // let context = RqPolyContext::new(fv.n, &fv.q);
//     // fv.context = Arc::new(context);
//     let a = fv.sample_uniform_poly();
//     let b = fv.sample_uniform_poly();
//     bench.iter(|| {
//         let _ = a.multiply_fast(&b);
//     })
// }

// fn bigint_ntt(bench: &mut Bencher) {
//     let fv = FV::<BigInt>::default_2048();
//     // let context = RqPolyContext::new(fv.n, &fv.q);
//     // fv.context = Arc::new(context);
//     let mut a = fv.sample_uniform_poly();
//     bench.iter(|| {
//         a.is_ntt_form = false;
//         let _ = a.forward_transform();
//     })
// }

// fn bigint_intt(bench: &mut Bencher) {
//     let fv = FV::<BigInt>::default_2048();
//     let mut a = fv.sample_uniform_poly();
//     bench.iter(|| {
//         a.is_ntt_form = true;
//         let _ = a.inverse_transform();
//     })
// }

// fn sample_uniform(bench: &mut Bencher) {
//     let fv = FV::<Scalar>::default_2048();

//     bench.iter(|| {
//         let _ = randutils::sample_uniform_poly(fv.context.clone());
//     })
// }

// fn sample_gaussian(bench: &mut Bencher) {
//     let fv = FV::<Scalar>::default_2048();

//     bench.iter(|| {
//         let _ = fv.sample_gaussian_poly(fv.stdev);
//     })
// }

// fn sample_uniform_scalar(bench: &mut Bencher) {
//     let fv = FV::<Scalar>::default_2048();

//     bench.iter(|| {
//         let _ = Scalar::sample_blw(&fv.q);
//     })
// }

// fn sample_binary(bench: &mut Bencher) {
//     let fv = FV::<Scalar>::default_2048();

//     bench.iter(|| {
//         let _ = fv.sample_binary_poly();
//     })
// }

// fn sample_binary_prng(bench: &mut Bencher) {
//     let fv = FV::<Scalar>::default_2048();

//     bench.iter(|| {
//         let _ = fv.sample_binary_poly_prng();
//     })
// }

// fn sample_uniform_scalar_from_rng(bench: &mut Bencher) {
//     let fv = FV::<Scalar>::default_2048();
//     let mut rng = StdRng::from_entropy();
//     bench.iter(|| {
//         let _ = Scalar::sample_below_from_rng(&fv.q, &mut rng);
//     })
// }

// fn sample_from_rng(bench: &mut Bencher) {
//     let fv = FV::<Scalar>::default_2048();
//     let mut rng = StdRng::from_entropy();
//     bench.iter(|| {
//         let _ = Scalar::_sample_form_rng(fv.q.bit_count, &mut rng);
//     })
// }

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
        let _ = fv.decrypt(&ct, &sk);
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

// benchmark_group!(
//     scalarop,
//     sample_uniform_scalar,
//     sample_uniform_scalar_from_rng
// );
// benchmark_group!(
//     polyop,
//     sample_uniform,
//     sample_gaussian,
//     sample_binary,
//     scalar_ntt,
//     scalar_intt,
//     sample_binary_prng
// );
benchmark_group!(
    fvop,
    encrypt_sk,
    encrypt_pk,
    encrypt_zero_pk,
    decryption,
    homomorphic_addition,
    rerandomize,
);

benchmark_main!(fvop);
