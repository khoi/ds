#[test]
fn generates_time_ordered_uuid_v7_values() {
    let timestamp = 1_742_000_123_456;

    let first = ds_ai::uuid_v7_at(timestamp).unwrap();
    let second = ds_ai::uuid_v7_at(timestamp).unwrap();

    assert_eq!(first.len(), 36);
    assert_eq!(&first[8..9], "-");
    assert_eq!(&first[13..14], "-");
    assert_eq!(&first[18..19], "-");
    assert_eq!(&first[23..24], "-");
    assert_eq!(&first[14..15], "7");
    assert!(matches!(&first[19..20], "8" | "9" | "a" | "b"));
    assert_eq!(uuid_timestamp(&first), timestamp);
    assert!(first < second);
}

#[test]
fn accepts_uuid_v7_timestamp_boundaries_and_rejects_overflow() {
    assert_eq!(uuid_timestamp(&ds_ai::uuid_v7_at(0).unwrap()), 0);
    assert_eq!(
        uuid_timestamp(&ds_ai::uuid_v7_at(0xffff_ffff_ffff).unwrap()),
        0xffff_ffff_ffff
    );
    assert!(matches!(
        ds_ai::uuid_v7_at(0x1_0000_0000_0000),
        Err(ds_ai::UuidV7Error::InvalidTimestamp)
    ));
}

fn uuid_timestamp(value: &str) -> u64 {
    u64::from_str_radix(&format!("{}{}", &value[..8], &value[9..13]), 16).unwrap()
}
