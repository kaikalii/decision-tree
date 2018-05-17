## Description

This is my implementation of a simple decision tree builder and evaluator, written in Rust.

After building with `cargo build`, type
```
cargo run -- -h
```
to view the command line options.

## Example Usage

There are two example data sets included with this project: `cars` and `mushrooms`. descriptions of the data sets can be found in their corresponding `*_info.txt` files. I will use the cars data set in this example.

To build a decision tree from the data:

```
cargo run examples/cars -b
```

To test the decision tree on a larger subset of the data:

```
cargo run examples/cars -t
```

To evaluate the outcomes of new data (there are two custom data points in `cars.txt`):

```
cargo run examples/cars -e
```

Using a combination of these flags will build the tree, test it, and evaluate new data, in that order:
```
cargo run examples/cars -b -t -e
```
