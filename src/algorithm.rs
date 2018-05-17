use std::{collections::{HashMap, HashSet},
          fmt};

// Calculates the entropy of a data set given the data and a set of possible outcomes
pub fn entropy(data: &Vec<(Vec<String>, String)>, outcomes: &HashSet<String>) -> f64 {
    let mut result = 0.0;
    for outcome in outcomes {
        let count = data.iter().filter(|x| &x.1 == outcome).count() as f64;
        if count > 0.0 {
            result += -count * count.log(2.0);
        }
    }
    result
}

pub fn information_gain(
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
pub fn id3(
    node: &mut Node,
    data: &Vec<(Vec<String>, String)>,
    variants: &Vec<HashSet<String>>,
    outcomes: &HashSet<String>,
    depth: usize,
    prune: Option<usize>,
    verbose: bool,
) {
    let get_common_coutcome = || -> String {
        let mut outcome_counts: HashMap<String, usize> =
            outcomes.iter().map(|x| (x.clone(), 0)).collect();
        for (_, o) in data {
            *outcome_counts.get_mut(o).unwrap() += 1;
        }
        outcome_counts.into_iter().max_by_key(|p| p.1).unwrap().0
    };
    // Check if the node is a leaf node
    if node.used_attrs.len() == variants.len() {
        node.outcome = Some(data.first().expect("outcome data is empty").1.clone());
        return;
    }
    // Check if pruning is necessary
    if let Some(prune_depth) = prune {
        if depth == prune_depth {
            node.outcome = Some(get_common_coutcome());
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
        if verbose {
            println!(
                "{}{}",
                (0..(2 * depth)).map(|_| ' ').collect::<String>(),
                best_attr
            );
        }
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
                id3(
                    &mut new_node,
                    &subset,
                    variants,
                    outcomes,
                    depth + 1,
                    prune,
                    verbose,
                );
            }
            node.children.push((variant.clone(), new_node));
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Node {
    pub attr: Option<usize>,
    pub children: Vec<(String, Node)>,
    #[serde(skip_serializing, skip_deserializing)]
    pub used_attrs: HashSet<usize>,
    pub outcome: Option<String>,
}

impl Node {
    pub fn test(&self, attributes: &[String], outcome: &str) -> bool {
        if let Some(ref o) = self.outcome {
            o == outcome
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
    pub fn eval(&self, attributes: &[String]) -> String {
        if let Some(ref o) = self.outcome {
            o.clone()
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
                .eval(attributes)
        } else {
            panic!("Node has no outcome and no attribute")
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
