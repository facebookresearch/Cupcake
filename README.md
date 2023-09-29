# Cupcake: A Rust Homomorphic Encryption Library

Cupcake is a powerful Rust library for the additive version of the Fan-Vercauteren homomorphic encryption scheme. It provides robust encryption of vectors, enabling operations such as vector addition, subtraction, and ciphertext rerandomization.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Building from Source](#building-from-source)
- [Examples](#examples)
- [Documentation](#documentation)
- [Benchmarks and Tests](#benchmarks-and-tests)
- [Supported Parameters](#supported-parameters)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Requirements

Cupcake is compatible with the following platforms:
- Mac OS X
- Linux

## Installation
## Installation

Getting started with Cupcake is easy, even if you're new to Rust or Cargo. Follow these clear installation instructions:

**1. Install Rust:**

If you haven't already, you'll need to install Rust, the programming language Cupcake is built with. You can do this by following the instructions at [Rust's official website](https://www.rust-lang.org/tools/install). Rust's package manager, Cargo, will be installed automatically with Rust.

**2. Create a New Rust Project (Optional):**

If you're starting a new project, you can create a new Rust project folder. Navigate to your desired project directory and run:

```bash
cargo new my_cupcake_project
cd my_cupcake_project


To include Cupcake in your Rust project, simply add the following line to your `Cargo.toml` file:

Cupcake = "0.2.1"
```

## Building from source

```bash
git clone https://github.com/facebookresearch/Cupcake
cd cupcake
cargo build --release
```

## Examples

Several examples are included in `examples/<name>.rs`, and can be run via
`cargo run --example <name>`

## Documentation

Documentation on the API can be built from `cargo doc`.

## Benchmarks and Tests

To maintain the high standards of reliability and performance, Cupcake includes benchmarking and testing tools. These tools play a crucial role in the development and utilization of the library.

Benchmarks: Benchmarks help gauge the efficiency of Cupcake's operations. By running benchmarks, you can identify which operations are faster and optimize your cryptographic workflows accordingly. Efficient operations are vital in applications that require real-time processing, such as secure data analytics and privacy-preserving machine learning.

Execute benchmarks with: 
`cargo bench` 

Tests: Testing ensures the correctness and reliability of Cupcake's functionality. It verifies that cryptographic operations produce accurate results, making Cupcake a trustworthy choice for your encryption needs. Rigorous testing helps uncover and fix potential issues early in the development process, ensuring the security of your applications.

Run the tests with:

`cargo test`.

## Supported parameters

Currently, we provide only one set of secure parameter, namely `FV::<Scalar>::default_2048();`. This parameter set has an estimated security level of about 128 bits according
to the homomorphic encryption security standards [link](http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf). Use other parameters at your own risk! With the default parameter set, the plaintext type is vector of `u8` with a fixed length 2048.

While Cupcake offers flexibility to work with other parameter sets, it's important to exercise caution when doing so. Changing parameters can affect the security guarantees and performance characteristics of the encryption scheme. Always assess your specific requirements and consult cryptography experts if you consider using custom parameters.

Understanding the relevance of these parameters is essential for making informed decisions about the security and performance of your applications.

## References

- [The Fan-Vercauteren scheme](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.400.6346&rep=rep1&type=pdf)

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
Cupcake is MIT licensed, as found in the LICENSE file.
