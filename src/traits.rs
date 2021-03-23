// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
/// The trait for symmetric key encryption.
pub trait SKEncryption<CT, PT, SK> {
    /// Generate a secret key
    fn generate_key(&self) -> SK;

    /// Use the secret key to generate a fresh encryption of zero
    fn encrypt_zero_sk(&self, sk: &SK) -> CT;

    /// Encrypt a given plaintext
    fn encrypt_sk(&self, pt: &PT, sk: &SK) -> CT;

    /// Decrypt a ciphertext
    fn decrypt(&self, ct: &CT, sk: &SK) -> PT;
}

/// The trait for public key encryption.
pub trait PKEncryption<CT, PT, SK>: SKEncryption<CT, PT, SK> {
    /// Generate a (pk, sk) keypair
    fn generate_keypair(&self) -> (CT, SK);

    /// Generate a fresh encryption of the zero plaintext
    fn encrypt_zero(&self, pk: &CT) -> CT;

    /// Encrypt a given plaintext
    fn encrypt(&self, pt: &PT, pk: &CT) -> CT;
}

/// The trait for additive homomorphic encryption.
pub trait AdditiveHomomorphicScheme<CT, PT, SK>: SKEncryption<CT, PT, SK> {
    /// Add a ciphertext into another.
    fn add_inplace(&self, ct1: &mut CT, ct2: &CT);

    /// Add a plaintext into a ciphertext.
    fn add_plain_inplace(&self, ct1: &mut CT, pt: &PT);

    /// Rerandomize a ciphertext in-place. The resulting ciphertext will decrypt to the same
    /// plaintext, while being unlinkable to the input ciphertext.
    fn rerandomize(&self, ct: &mut CT, pk: &CT);
}
