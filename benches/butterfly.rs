// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#[macro_use]
extern crate bencher;
use bencher::Bencher;
use cupcake::integer_arith::scalar::Scalar;
use cupcake::integer_arith::butterfly::{butterfly, inverse_butterfly, lazy_butterfly, lazy_butterfly_u64};
use cupcake::integer_arith::ArithUtils;

#[allow(non_snake_case)]
fn bench_butterfly(bench: &mut Bencher) {
    
    let q = Scalar::new_modulus(18014398492704769u64);
    let x = rand::random::<u64>();
    let y = rand::random::<u64>();
    let w = rand::random::<u64>();


    let mut X = Scalar::from(x);
    let mut Y = Scalar::from(y);
    let W = Scalar::from(w); 

    bench.iter(|| {
        let _ = butterfly(&mut X, &mut Y, &W, &q); 
    })
}

#[allow(non_snake_case)]
fn bench_inverse_butterfly(bench: &mut Bencher) {
    
    let q = Scalar::new_modulus(18014398492704769u64);
    let x = rand::random::<u64>();
    let y = rand::random::<u64>();
    let w = rand::random::<u64>();


    let mut X = Scalar::from(x);
    let mut Y = Scalar::from(y);
    let W = Scalar::from(w); 

    bench.iter(|| {
        let _ = inverse_butterfly(&mut X, &mut Y, &W, &q); 
    })
}

#[allow(non_snake_case)]
fn bench_lazy_butterfly(bench: &mut Bencher) {
    
    let q = Scalar::new_modulus(18014398492704769u64);
    let x = rand::random::<u64>();
    let y = rand::random::<u64>();
    let w = rand::random::<u64>();


    let mut X = Scalar::from(x);
    let mut Y = Scalar::from(y);
    let W = Scalar::from(w); 

    let Wprime: u64 = cupcake::integer_arith::util::compute_harvey_ratio(W.rep(), q.rep());

    let twoq: u64 = q.rep() << 1; 

    bench.iter(|| {
        let _ = lazy_butterfly_u64(x, y, W.rep(), Wprime,  q.rep(), twoq); 
    })
}

benchmark_group!(
    butterfly_group,
    bench_butterfly,
    bench_inverse_butterfly,
    bench_lazy_butterfly
);


benchmark_main!(butterfly_group);
