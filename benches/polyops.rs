// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#[macro_use]
extern crate bencher;
use bencher::Bencher;
use cupcake::rqpoly::RqPolyContext;
use cupcake::traits::*;
pub use std::sync::Arc;
use cupcake::integer_arith::scalar::Scalar;
use cupcake::integer_arith::ArithUtils;
use cupcake::randutils;

fn scalar_ntt(bench: &mut Bencher) {
    let q = Scalar::new_modulus(18014398492704769u64);
    let context = Arc::new(RqPolyContext::new(2048, &q));
    let mut testpoly = randutils::sample_uniform_poly(context);

    bench.iter(|| {
        testpoly.is_ntt_form = false;
        let _ = testpoly.forward_transform();
    })
}

fn scalar_intt(bench: &mut Bencher) {
    let q = Scalar::new_modulus(18014398492704769u64);
    let context = Arc::new(RqPolyContext::new(2048, &q));

    let mut testpoly = cupcake::randutils::sample_uniform_poly(context.clone());

    bench.iter(|| {
        testpoly.is_ntt_form = true;
        let _ = testpoly.inverse_transform();
    })
}

fn sample_uniform(bench: &mut Bencher) {
    let q = Scalar::new_modulus(18014398492704769u64);
    let context = Arc::new(RqPolyContext::new(2048, &q));

    bench.iter(|| {
        let _ = randutils::sample_uniform_poly(context.clone());
    })
}

fn sample_gaussian(bench: &mut Bencher) {
    let q = Scalar::new_modulus(18014398492704769u64);
    let context = Arc::new(RqPolyContext::new(2048, &q));

    bench.iter(|| {
        let _ = randutils::sample_gaussian_poly(context.clone(), 3.14);
    })
}

fn sample_ternary(bench: &mut Bencher) {
    let q = Scalar::new_modulus(18014398492704769u64);
    let context = Arc::new(RqPolyContext::new(2048, &q));

    bench.iter(|| {
        let _ = randutils::sample_ternary_poly_prng(context.clone());
    })
}

benchmark_group!(
    polyops,
    sample_gaussian,
    sample_ternary,
    sample_uniform,
    scalar_ntt,
    scalar_intt,
);

benchmark_main!(polyops);
