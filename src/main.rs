extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate rand;

mod algorithm;

use std::{collections::HashSet, env, fs::File, io::Read, path::PathBuf};

use rand::{thread_rng, Rng};

use algorithm::*;

fn main() {
    let mut rng = thread_rng();
    let mut args = env::args();
    args.next();
    let mut file_stem_set = false;
    let mut file_stem;
    let mut json_file = PathBuf::new();
    let mut data_file = PathBuf::new();
    let mut txt_file = PathBuf::new();
    let mut build = false;
    let mut test = false;
    let mut eval = false;
    let mut prune = None;
    let mut ratio = 0.2;
    let mut first = false;
    let mut verbose = false;
    let mut nosave = false;
    while let Some(arg) = args.next() {
        match arg.as_ref() {
            "-b" | "--build" => build = true,
            "-t" | "--test" => test = true,
            "-e" | "--eval" => eval = true,
            "-p" | "--prune" => {
                prune = args.next()
                    .map(|x| Some(x.parse::<usize>().expect("Invalid prune depth")))
                    .unwrap_or(None)
            }
            "-r" | "--ratio" => {
                ratio = args.next()
                    .map(|x| x.parse::<f64>().expect("Invalid test ratio"))
                    .unwrap_or(0.2)
            }
            "-f" | "--first" => first = true,
            "--verbose" => verbose = true,
            "--nosave" => nosave = true,
            "-h" | "--help" => {
                println!(
                    "
usage:
    decision-tree [data name] <flags>

flags:
    -b | --build        Builds the tree from a subset of the data
    -t | --test         Tests the tree on a subset of the data
    -e | --eval         Evaluate news data
    -p | --prune        Sets the maximum depth of the tree
    -r | --ratio        Sets the ratio of training data to total data
    -f | --first        Makes the first element in each row of data
                        be read as the outcome, rather than the last
    --verbose           When used with -b | --build, prints the
                        indices of the attributes of each node
    --nosave            Prevents the tree from saving to a file
    -h | --help         Prints this message
                "
                );
                return;
            }
            _ => {
                file_stem = arg;
                json_file = PathBuf::from(file_stem.clone()).with_extension("json");
                data_file = PathBuf::from(file_stem.clone()).with_extension("data");
                txt_file = PathBuf::from(file_stem.clone()).with_extension("txt");
                file_stem_set = true;
            }
        }
    }

    if !file_stem_set {
        println!("Expected data file path stem");
        return;
    }

    // Load the data
    let mut data_bytes = Vec::new();
    File::open(data_file)
        .expect("unable to open data file")
        .read_to_end(&mut data_bytes)
        .expect("Unable to read data file to string");
    let full_string = String::from_utf8_lossy(&data_bytes);
    let data_strings: Vec<String> = full_string
        .split_whitespace()
        .map(|x| x.to_string())
        .collect();
    let count = (data_strings.len() as f64 * ratio) as usize;

    // Read the data from the file into a vector
    let data: Vec<(Vec<String>, String)> = data_strings
        .into_iter()
        .map(|entry| {
            let mut attributes: Vec<String> = entry.split(",").map(|x| x.to_string()).collect();
            if first {
                (
                    attributes.iter().skip(1).map(|x| x.to_string()).collect(),
                    attributes[0].to_string(),
                )
            } else {
                let outcome = attributes.pop().unwrap();
                (attributes, outcome)
            }
        })
        .collect();

    // Load a subset of the data
    let mut training_data: Vec<(Vec<String>, String)> = Vec::new();
    let mut used_entries = HashSet::new();
    while used_entries.len() < count {
        let random_entry = rng.gen_range(0, data.len());
        if !used_entries.contains(&random_entry) {
            training_data.push(data[random_entry].clone());
            used_entries.insert(random_entry);
        }
    }
    println!("training data size: {}", training_data.len());
    let mut test_data: Vec<(Vec<String>, String)> = Vec::new();
    for (i, entry) in data.iter().enumerate() {
        if !used_entries.contains(&i) {
            test_data.push(entry.clone());
        }
    }
    println!("test data size: {}", test_data.len());

    // Collect the different possible attribute variants and outcomes
    let mut variants = vec![HashSet::new(); data.first().expect("data is empty").0.len()];
    let mut outcomes = HashSet::new();
    for (entry, outcome) in &data {
        for (i, attr) in entry.iter().enumerate() {
            variants[i].insert(attr.clone());
        }
        outcomes.insert(outcome.clone());
    }
    println!("variants: {:?}", variants);
    println!("outcomes: {:?}", outcomes);

    let mut root = None;

    // Building a tree
    if build {
        println!("Building...");

        // Create the root node
        root = Some(Node::default());

        // Run the algorithm
        id3(
            root.as_mut().unwrap(),
            &training_data,
            &variants,
            &outcomes,
            0,
            prune,
            verbose,
        );
        println!("Tree construction complete");

        // Save the tree
        if !nosave {
            let out_file = File::create(json_file.clone()).expect("Unable to create output file");
            serde_json::to_writer_pretty(out_file, &root).expect("Unable to serialize tree");
            println!("Tree saved to file");
        }

        // Test the training data
        let (mut successes, mut failures) = (0, 0);
        for entry in &training_data {
            if root.as_mut().unwrap().test(&entry.0, &entry.1) {
                successes += 1;
            } else {
                failures += 1;
            }
        }
        println!("..............................");

        // Report results
        println!(
            "training data:\n    {} successes, {} failures\n    {}% accuracy",
            successes,
            failures,
            (successes as f32) * 100.0 / (successes + failures) as f32
        );
        println!("..............................");
    }
    // Build the tree from the tree file if it was not built
    if root.is_none() {
        root = Some(
            serde_json::from_reader(
                File::open(json_file.clone()).expect("unable to open tree file"),
            ).expect("Unable to deserialize tree file"),
        );
    }
    // Testing a tree
    if test {
        // Test the test data
        let (mut successes, mut failures) = (0, 0);
        for entry in &test_data {
            if root.as_mut().unwrap().test(&entry.0, &entry.1) {
                successes += 1;
            } else {
                failures += 1;
            }
        }

        // Report results
        println!(
            "test_data:\n    {} successes, {} failures\n    {}% accuracy",
            successes,
            failures,
            (successes as f32) * 100.0 / (successes + failures) as f32
        );
        println!("..............................");
    }
    // Evaluate new data
    if eval {
        // Load the eval data
        let mut eval_data_bytes = Vec::new();
        File::open(txt_file)
            .expect("unable to open eval file")
            .read_to_end(&mut eval_data_bytes)
            .expect("Unable to read eval file to string");
        let eval_full_string = String::from_utf8_lossy(&eval_data_bytes);
        let eval_data_strings: Vec<String> = eval_full_string
            .split_whitespace()
            .map(|x| x.to_string())
            .collect();

        // Read the eval data from the file into a vector
        let eval_data: Vec<Vec<String>> = eval_data_strings
            .into_iter()
            .map(|entry| entry.split(",").map(|x| x.to_string()).collect())
            .collect();

        // Evaluate the data
        println!("Evaluation:");
        for entry in eval_data {
            println!("{:?}: {}", entry, root.as_mut().unwrap().eval(&entry));
        }
        println!("..............................");
    }
}
