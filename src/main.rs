extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate rand;

use std::{collections::{HashMap, HashSet},
          env,
          fmt,
          fs::File,
          io::Read,
          path::PathBuf};

use rand::{thread_rng, Rng};

fn main() {
    let mut rng = thread_rng();
    let mut args = env::args();
    args.next();
    let file_name = PathBuf::from(
        args.next()
            .expect("Expected training data / tree file name parameter"),
    );
    let ratio = args.next()
        .map(|x| x.parse::<f64>().expect("Invalid test ratio"))
        .unwrap_or(0.2);
    let first = args.next().map(|x| x == "first").unwrap_or(false);

    // Testing a tree
    if file_name.extension().expect("file has no extension") == "json" {
        let test_file_name =
            PathBuf::from(args.next().expect("Expected test data file name parameter"));
        let mut test_file =
            File::open(test_file_name.clone()).expect("Unable to open the data file.");
        let tree_file = File::open(file_name).expect("Unable to open tree file");

        let mut data_bytes = Vec::new();
        test_file
            .read_to_end(&mut data_bytes)
            .expect("Unable to read data file to string");
        let full_string = String::from_utf8_lossy(&data_bytes);
        let data_strings: Vec<String> = full_string
            .split_whitespace()
            .map(|x| x.to_string())
            .collect();
        let count = (data_strings.len() as f64 * ratio) as usize;

        let root: Node =
            serde_json::from_reader(tree_file).expect("Unable to deserialize tree file");

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
        // Load subsets of the data
        let mut test_data: Vec<(Vec<String>, String)> = Vec::new();
        for _ in 0..(data.len() - count) {
            test_data.push(data[rng.gen_range(0, data.len())].clone());
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

        let (mut successes, mut failures) = (0, 0);
        for (i, entry) in test_data.iter().enumerate() {
            if i % (test_data.len() / 30) == 0 {
                print!(".");
            }
            if root.test(&entry.0, &entry.1) {
                successes += 1;
            } else {
                failures += 1;
            }
        }
        println!();
        println!(
            "test_data:\n    {} successes, {} failures\n    {}% accuracy",
            successes,
            failures,
            (successes as f32) * 100.0 / (successes + failures) as f32
        );

    // Building a tree
    } else {
        let prune = args.next()
            .map(|x| Some(x.parse::<usize>().expect("Invalid prune depth")))
            .unwrap_or(None);
        let mut data_file = File::open(file_name.clone()).expect("Unable to open the data file.");

        let mut data_bytes = Vec::new();
        data_file
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
        for _ in 0..count {
            training_data.push(data[rng.gen_range(0, data.len())].clone());
        }
        println!("training data size: {}", training_data.len());

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

        // Create the root node
        let mut root = Node::default();

        // Run the algorithm
        id3(&mut root, &training_data, &variants, &outcomes, 0, prune);
        println!("Tree construction complete");

        // Save the tree
        let out_file =
            File::create(file_name.with_extension("json")).expect("Unable to create output file");
        serde_json::to_writer_pretty(out_file, &root).expect("Unable to serialize tree");
        println!("Tree saved to file");

        // Test the training data
        let (mut successes, mut failures) = (0, 0);
        for (i, entry) in training_data.iter().enumerate() {
            if i % (training_data.len() / 30) == 0 {
                print!(".");
            }
            if root.test(&entry.0, &entry.1) {
                successes += 1;
            } else {
                failures += 1;
            }
        }
        println!();
        println!(
            "training data:\n    {} successes, {} failures\n    {}% accuracy",
            successes,
            failures,
            (successes as f32) * 100.0 / (successes + failures) as f32
        );
    }
}

// Calculates the entropy of a data set given the data and a set of possible outcomes
fn entropy(data: &Vec<(Vec<String>, String)>, outcomes: &HashSet<String>) -> f64 {
    let mut result = 0.0;
    for outcome in outcomes {
        let count = data.iter().filter(|x| &x.1 == outcome).count() as f64;
        if count > 0.0 {
            result += -count * count.log(2.0);
        }
    }
    result
}

fn information_gain(
    data: &Vec<(Vec<String>, String)>,
    variants: &Vec<HashSet<String>>,
    outcomes: &HashSet<String>,
    attr_index: usize,
) -> f64 {
    let mut sum = 0.0;
    for variant in &variants[attr_index] {
        let subset: Vec<(Vec<String>, String)> = data.iter()
            .filter(|(entry, _)| &entry[attr_index] == variant)
            .cloned()
            .collect();
        sum += subset.len() as f64 / data.len() as f64 * entropy(&subset, outcomes);
    }

    entropy(data, outcomes) - sum
}

// Runs the ID3 algorithm to build the tree
fn id3(
    node: &mut Node,
    data: &Vec<(Vec<String>, String)>,
    variants: &Vec<HashSet<String>>,
    outcomes: &HashSet<String>,
    depth: usize,
    prune: Option<usize>,
) {
    // Check if the node is a leaf node
    if entropy(data, outcomes) == 0.0 {
        node.outcome = Some(data.first().expect("outcome data is empty").1.clone());
        return;
    }
    // Check if pruning is necessary
    if let Some(prune_depth) = prune {
        if depth == prune_depth {
            let mut outcome_counts: HashMap<String, usize> =
                outcomes.iter().map(|x| (x.clone(), 0)).collect();
            for (_, o) in data {
                *outcome_counts.get_mut(o).unwrap() += 1;
            }
            node.outcome = Some(outcome_counts.into_iter().max_by_key(|p| p.1).unwrap().0);
            return;
        }
    }
    // Iterate through unused attributes and find the one with the max entropy
    let best_attr = (0..(data.first().map(|x| x.0.len()).unwrap_or(0)))
        .filter(|i| !node.used_attrs.contains(i))
        .max_by(|a, b| {
            information_gain(data, variants, outcomes, *a)
                .partial_cmp(&information_gain(data, variants, outcomes, *b))
                .expect("partial_cmp is none")
        });

    if let Some(best_attr) = best_attr {
        println!(
            "{}{}",
            (0..(2 * depth)).map(|_| ' ').collect::<String>(),
            best_attr
        );
        node.attr = Some(best_attr);
        node.used_attrs.insert(best_attr);

        for variant in &variants[best_attr] {
            let mut new_node = Node::default();
            new_node.used_attrs = node.used_attrs.clone();
            let subset: Vec<(Vec<String>, String)> = data.iter()
                .filter(|(entry, _)| &entry[best_attr] == variant)
                .cloned()
                .collect();
            if subset.is_empty() {
                node.outcome = Some(data.first().expect("data is empty").1.clone());
            } else {
                id3(&mut new_node, &subset, variants, outcomes, depth + 1, prune);
            }
            node.children.push((variant.clone(), new_node));
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct Node {
    pub attr: Option<usize>,
    pub children: Vec<(String, Node)>,
    pub used_attrs: HashSet<usize>,
    pub outcome: Option<String>,
}

impl Node {
    pub fn test(&self, attributes: &[String], outcome: &str) -> bool {
        if let Some(ref o) = self.outcome {
            return o == outcome;
        } else if let Some(attr_index) = self.attr {
            let my_variant = attributes[attr_index].clone();
            self.children
                .iter()
                .find(|v| v.0 == my_variant)
                .expect(&format!(
                    "node has no variant that matches the attribute: {}",
                    my_variant
                ))
                .1
                .test(attributes, outcome)
        } else {
            false
        }
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "(attribute: {:?}, children: {:#?}, {})",
            self.attr,
            self.children,
            if let Some(ref s) = self.outcome {
                s
            } else {
                ""
            }
        )
    }
}
