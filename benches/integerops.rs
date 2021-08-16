// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#[macro_use]
extern crate bencher;
use bencher::Bencher;
use cupcake::integer_arith::scalar::Scalar;
use cupcake::integer_arith::ArithUtils;

#[allow(non_snake_case)]
fn bench_mulmod(bench: &mut Bencher) {
    
    let q = Scalar::new_modulus(18014398492704769u64);
    let x = rand::random::<u64>();
    let y = rand::random::<u64>();

    let X = Scalar::from(x);
    let Y = Scalar::from(y);

    bench.iter(|| {
        let _ = Scalar::mul_mod(&X, &Y, &q); 
    })
}


benchmark_group!(
    integerops_group,
    bench_mulmod,
);


benchmark_main!(integerops_group);
