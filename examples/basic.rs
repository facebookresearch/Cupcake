// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use cupcake::traits::*;

fn smartprint<T: std::fmt::Debug>(v: &Vec<T>) {
    println!("[{:?}, {:?}, ..., {:?}]", v[0], v[1], v[v.len() - 1]);
}

fn main() {
    let fv = cupcake::default();

    let (pk, sk) = fv.generate_keypair();

    print!("Encrypting a constant vector v of 1s...");
    let v = vec![1; fv.n];

    let mut ctv = fv.encrypt(&v, &pk);

    let mut pt_actual: Vec<u8> = fv.decrypt(&ctv, &sk);
    print!("decrypted v: ");
    smartprint(&pt_actual);

    print!("Encrypting a constant vector w of 2s...");
    let w = vec![2; fv.n];
    let ctw = fv.encrypt(&w, &pk);

    pt_actual = fv.decrypt(&ctw, &sk);
    print!("decrypted w: ");
    smartprint(&pt_actual);

    // add ctw into ctv
    fv.add_inplace(&mut ctv, &ctw);
    print!("Decrypting the sum...");
    pt_actual = fv.decrypt(&ctv, &sk);
    print!("decrypted v+w: ");
    smartprint(&pt_actual);

    // add the plaintext w into the ciphertext
    fv.add_plain_inplace(&mut ctv, &w);
    print!("Decrypting the sum...");
    pt_actual = fv.decrypt(&ctv, &sk);
    print!("decrypted v+w+w: ");
    smartprint(&pt_actual);
}
