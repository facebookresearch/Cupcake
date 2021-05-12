// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use cupcake::traits::{AdditiveHomomorphicScheme, PKEncryption, SKEncryption};

fn smartprint<T: std::fmt::Debug>(v: &Vec<T>) {
    println!("[{:?}, {:?}, ..., {:?}]", v[0], v[1], v[v.len() - 1]);
}

fn main() {
    let fv = cupcake::default();

    let (pk, sk) = fv.generate_keypair();

    println!("Encrypting a vector [0,1,2,3,...]");
    let mut v = vec![];
    for i in 0..fv.n {
        v.push(i as u8);
    }

    let mut ctv = fv.encrypt(&v, &pk);

    let pt_original = fv.decrypt(&ctv, &sk);
    print!("decrypted value: ");
    smartprint(&pt_original);

    println!("Rerandomizing the ciphertext...");

    fv.rerandomize(&mut ctv, &pk);
    print!("decrypted value after reranromization: ");
    let pt_new = fv.decrypt(&ctv, &sk);
    smartprint(&pt_new);

    print!("Check that the plaintext has not changed...");
    assert_eq!(pt_original, pt_new);
    println!("ok");
}
