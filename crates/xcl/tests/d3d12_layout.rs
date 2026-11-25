//! Invariants over the generated D3D12 interface table (`hle/d3d12_layout.rs`) and its
//! lookup helpers (`hle/d3d12x/layout.rs`). A generator regression (bad IID, duplicated
//! or missing method, a non-versioned sibling leaking into a chain) would silently
//! misalign slots in the D3D12.X device shim.

use xcl::hle::d3d12_layout::IFACES;
use xcl::hle::d3d12x::layout::{chain, find, guid_string, parse_guid, slot};

fn numbered_version_filter(name: &str, base: &str) -> bool {
    name.starts_with(base) && name[base.len()..].chars().all(|c| c.is_ascii_digit())
}

#[test]
fn table_shape_and_unique_ids() {
    assert!(!IFACES.is_empty(), "generated table is empty");

    let mut names = std::collections::HashSet::new();
    let mut iids = std::collections::HashSet::new();
    for iface in IFACES {
        assert!(!iface.name.is_empty(), "entry with empty name");
        assert!(names.insert(iface.name), "duplicate interface name {}", iface.name);
        assert!(!iface.iid.is_empty(), "{} has empty iid", iface.name);
        assert!(iids.insert(iface.iid), "duplicate iid {} ({})", iface.iid, iface.name);
        assert!(!iface.parent.is_empty(), "{} has empty parent", iface.name);
        assert!(iface.methods.len() >= 3, "{} has too few methods", iface.name);
        assert_eq!(
            iface.methods[..3],
            ["QueryInterface", "AddRef", "Release"],
            "{} must start with the IUnknown vtable head",
            iface.name
        );
    }
}

#[test]
fn iid_round_trips_through_parse_guid() {
    for iface in IFACES {
        assert_eq!(guid_string(&parse_guid(iface.iid)), iface.iid, "iid round-trip for {}", iface.name);
    }

    // canonical layout: 8-4-4-4-12 hex groups, 36 chars, lowercase
    let s = guid_string(&parse_guid(IFACES[0].iid));
    assert_eq!(s.len(), 36);
    assert_eq!(s.chars().filter(|&c| c == '-').count(), 4);
    assert_eq!(&s[8..9], "-");
    assert_eq!(&s[13..14], "-");
    assert_eq!(&s[18..19], "-");
    assert_eq!(&s[23..24], "-");
    assert!(s.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));

    // non-hex text must not panic or produce garbage; it yields the all-zero GUID
    for bad in ["zzzz", "!!!", ""] {
        assert_eq!(guid_string(&parse_guid(bad)), "00000000-0000-0000-0000-000000000000");
    }
    assert_eq!(guid_string(&parse_guid("zzzz")), guid_string(&parse_guid("!!!")));
}

#[test]
fn device_chain_is_base_plus_numbered_versions_most_derived_first() {
    let base = "ID3D12Device";
    let expected = IFACES.iter().filter(|i| numbered_version_filter(i.name, base)).count();
    assert!(expected >= 10, "expected the full ID3D12Device chain, got {expected}");

    let c = chain(base);
    assert_eq!(c.len(), expected, "chain must hold the base plus every numbered descendant");
    assert!(c.iter().any(|i| i.name == base), "chain must include the base");
    assert!(c.iter().all(|i| i.name.starts_with(base)));
    assert!(!c.iter().any(|i| i.name == "ID3D12DeviceChild"), "non-versioned sibling leaked into the chain");
    assert!(
        !c.iter().any(|i| i.name == "ID3D12DeviceFactory" || i.name == "ID3D12DeviceConfiguration"),
        "non-versioned siblings leaked into the chain"
    );
    for w in c.windows(2) {
        assert!(w[0].methods.len() >= w[1].methods.len(), "chain must be most-derived-first by method count");
    }
    assert_eq!(c[0].methods.len(), c.iter().map(|i| i.methods.len()).max().unwrap());
}

#[test]
fn command_queue_chain_is_most_derived_first() {
    let base = "ID3D12CommandQueue";
    let c = chain(base);
    assert_eq!(c.len(), IFACES.iter().filter(|i| numbered_version_filter(i.name, base)).count());
    assert_eq!(c[0].name, "ID3D12CommandQueue1");
    assert!(c.iter().all(|i| i.name.starts_with(base)));
    for w in c.windows(2) {
        assert!(w[0].methods.len() >= w[1].methods.len());
    }
}

#[test]
fn slot_returns_known_indices_and_rejects_wrong_names() {
    let dev = find("ID3D12Device").unwrap();
    assert_eq!(slot(dev, "QueryInterface"), Some(0));
    assert_eq!(slot(dev, "CreateCommandQueue"), Some(8));
    assert_eq!(slot(dev, "CreateCommittedResource"), Some(27));
    assert_eq!(slot(dev, "CreateFence"), Some(36));
    let queue = find("ID3D12CommandQueue").unwrap();
    assert_eq!(slot(queue, "ExecuteCommandLists"), Some(10));

    // unknown and misspelled methods must not resolve
    assert_eq!(slot(dev, "CreateCommandQueu"), None);
    assert_eq!(slot(dev, "NonexistentMethod"), None);
    assert_eq!(slot(dev, ""), None);
}

#[test]
fn find_slot_and_chain_are_consistent() {
    assert!(find("ID3D12DoesNotExist").is_none(), "find must miss unknown interfaces");
    assert!(find("").is_none(), "find must miss the empty name");

    for iface in IFACES {
        assert!(find(iface.name).is_some_and(|found| found.name == iface.name && found.iid == iface.iid && found.parent == iface.parent && found.methods == iface.methods), "find must return the table entry for {}", iface.name);
    }

    // every numbered version resolves via find and is listed by its parent's chain
    let versions: Vec<_> = IFACES
        .iter()
        .filter(|i| numbered_version_filter(i.name, "ID3D12Device") && i.name != "ID3D12Device")
        .collect();
    assert!(!versions.is_empty());
    for version in versions {
        let parent = find(version.parent).unwrap_or_else(|| panic!("parent {} of {} is missing", version.parent, version.name));
        // the base chain lists every numbered descendant of the base
        assert!(chain("ID3D12Device").iter().any(|i| i.name == version.name), "chain(ID3D12Device) must list {}", version.name);
        assert!(chain(version.name).iter().any(|i| i.name == version.name), "chain({}) must list itself", version.name);

        // an inherited method keeps the same slot in the child and its parent
        let idx = slot(version, "CreateCommandQueue").unwrap();
        assert_eq!(slot(parent, "CreateCommandQueue"), Some(idx));
        assert_eq!(version.methods[idx], "CreateCommandQueue");
    }
}
