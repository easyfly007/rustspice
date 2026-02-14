use std::fs;
use std::path::{Path, PathBuf};

fn collect_files(root: &Path, limit: usize, out: &mut Vec<PathBuf>) {
    if out.len() >= limit {
        return;
    }
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            if out.len() >= limit {
                break;
            }
            let path = entry.path();
            if path.is_dir() {
                collect_files(&path, limit, out);
            } else {
                out.push(path);
            }
        }
    }
}

#[test]
fn spice_datasets_is_accessible() {
    let dataset_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("spice-datasets");
    assert!(
        dataset_root.exists(),
        "spice-datasets directory not found at {:?}",
        dataset_root
    );

    let mut files = Vec::new();
    collect_files(&dataset_root, 3, &mut files);
    assert!(
        !files.is_empty(),
        "spice-datasets does not contain any files"
    );

    for file in files {
        let content = fs::read(&file).unwrap_or_default();
        assert!(
            !content.is_empty(),
            "dataset file is empty: {:?}",
            file
        );
    }
}
