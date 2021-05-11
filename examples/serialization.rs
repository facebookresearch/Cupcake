// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use cupcake::traits::*;
use cupcake::FVCiphertext;
fn smartprint<T: std::fmt::Debug>(v: &Vec<T>) {
    println!("[{:?}, {:?}, ..., {:?}]", v[0], v[1], v[v.len() - 1]);
}

fn main() {
    let fv = cupcake::default();

    let (pk, sk) = fv.generate_keypair();

    print!("Encrypting a constant vector v of 1s...");
    let v = vec![1; fv.n];

    let ctv = fv.encrypt(&v, &pk);

    print!("Encrypting a constant vector w of 2s...");
    let w = vec![2; fv.n];
    let ctw = fv.encrypt(&w, &pk);

    // serialize
    print!("Serializing the ciphertexts...");

    let serialized_ctv = ctv.to_bytes();
    let serialized_ctw = ctw.to_bytes();

    // deserializing
    let mut deserialized_ctv = FVCiphertext::from_bytes(&serialized_ctv);
    let mut deserialized_ctw = FVCiphertext::from_bytes(&serialized_ctw);

    // add ctw into ctv
    print!("Adding the deserialized ciphertexts...");
    fv.set_context(&mut deserialized_ctv);
    fv.set_context(&mut deserialized_ctw);

    fv.add_inplace(&mut deserialized_ctv, &deserialized_ctw);
    print!("Decrypting the sum...");
    let pt_actual = fv.decrypt(&deserialized_ctv, &sk);
    print!("decrypted v+w: ");
    smartprint(&pt_actual);
}
