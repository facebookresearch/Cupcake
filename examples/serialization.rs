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

    println!("Encrypting a constant vector v of 1s...");
    let v = vec![1; fv.n];

    let ctv = fv.encrypt(&v, &pk);

    println!("Encrypting a constant vector w of 2s...");
    let w = vec![2; fv.n];
    let ctw = fv.encrypt(&w, &pk);

    // serialize
    println!("Serializing the ciphertexts...");

    let serialized_ctv = ctv.to_bytes();
    let serialized_ctw = ctw.to_bytes();

    // deserializing
    let mut deserialized_ctv = fv.from_bytes(&serialized_ctv);
    let deserialized_ctw = fv.from_bytes(&serialized_ctw);

    // add ctw into ctv
    println!("Adding the deserialized ciphertexts...");
    fv.add_inplace(&mut deserialized_ctv, &deserialized_ctw);
    println!("Decrypting the sum...");
    let pt_actual: Vec<u8> = fv.decrypt(&deserialized_ctv, &sk);
    println!("decrypted v+w: ");
    smartprint(&pt_actual);
}
