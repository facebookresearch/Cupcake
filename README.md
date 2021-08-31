# Cupcake

Cupcake is an efficient Rust library for the (additive version of) Fan-Vercauteren homomorphic encryption scheme, offering capabilities to
encrypt vectors, add/subtract two encrypted vectors, and rerandomize a ciphertext.


## Requirements
Cupcake requires or works with
* Mac OS X or Linux

## Installation
Add the following line to the dependencies of your Cargo.toml:
```
Cupcake = "0.1.1"
```

## Building from source

```bash
git clone https://github.com/facebookresearch/Cupcake
cd cupcake
cargo build --release
```

## Examples

Several examples are included in `examples/<name>.rs`, and can be run via
`cargo run --example <name>` or `cargo run --release --example <name>`

## Documentation

Documentation on the API can be built from `cargo doc`.

## Benchmarks and Tests

We have included benchmarks and tests for both homomorphic operations and underlying arithmetic operations. They can be run using `cargo bench` and `cargo test`. Additional tests can be run for the example CupcakeParallel wrapper with `cargo test --examples parallel`

## Supported parameters

Currently, we provide only one set of secure parameter, namely `FV::<Scalar>::default_2048();`. This parameter set has an estimated security level of about 128 bits according
to the homomorphic encryption security standards [link](http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf). Use other parameters at your own risk! With the default parameter set, the plaintext type is vector of `u8` with a fixed length 2048.


## References

- [The Fan-Vercauteren scheme](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.400.6346&rep=rep1&type=pdf)

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
Cupcake is MIT licensed, as found in the LICENSE file.
