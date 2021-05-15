// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
/// Trait for key generation
pub trait KeyGeneration<PK,SK>: EncryptionOfZeros<PK, SK>{
    /// Generate a (pk, sk) keypair
    fn generate_keypair(&self) -> (PK, SK);

    /// Generate a secret key
    fn generate_key(&self) -> SK;
}

/// Trait for encryption of zeros.
pub trait EncryptionOfZeros<CT, SK>{
    /// Use the secret key to generate a fresh encryption of zero
    fn encrypt_zero_sk(&self, sk: &SK) -> CT;

    /// Generate a fresh encryption of the zero plaintext
    fn encrypt_zero(&self, pk: &CT) -> CT;
}

/// Trait for symmetric key encryption.
pub trait SKEncryption<CT, PT, SK>: KeyGeneration<CT, SK>{

    /// Encrypt a given plaintext
    fn encrypt_sk(&self, pt: &PT, sk: &SK) -> CT;

    /// Decrypt a ciphertext
    fn decrypt(&self, ct: &CT, sk: &SK) -> PT;
}

/// Trait for public key encryption.
pub trait PKEncryption<CT, PT, SK>: SKEncryption<CT, PT, SK> {
    /// Encrypt a given plaintext
    fn encrypt(&self, pt: &PT, pk: &CT) -> CT;
}

pub trait CipherPlainAddition<CT, PT>: {
    /// Add a plaintext into a ciphertext.
    fn add_plain_inplace(&self, ct1: &mut CT, pt: &PT);
}


/// Trait for additive homomorphic operations.
pub trait AdditiveHomomorphicScheme<CT, SK>: EncryptionOfZeros<CT, SK> {
    /// Add a ciphertext into another.
    fn add_inplace(&self, ct1: &mut CT, ct2: &CT);

    /// Rerandomize a ciphertext in-place. The resulting ciphertext will decrypt to the same
    /// plaintext, while being unlinkable to the input ciphertext.
    fn rerandomize(&self, ct: &mut CT, pk: &CT);
}

pub trait Serializable{
    /// Serialize to a vector of bytes.
    fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize from a vector of bytes.
    fn from_bytes(bytes: &Vec<u8>) -> Self;
}
