use std::{
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

const MAX_TIMESTAMP: u64 = 0xffff_ffff_ffff;
const MAX_SEQUENCE: u64 = (1 << 41) - 1;

#[derive(Debug, Error)]
pub enum UuidV7Error {
    #[error("UUIDv7 timestamp must be between 0 and {MAX_TIMESTAMP}")]
    InvalidTimestamp,
    #[error("UUIDv7 generator sequence exhausted")]
    SequenceExhausted,
}

#[derive(Default)]
struct State {
    last_ordinary_timestamp: u64,
    sequence: Option<u64>,
}

pub fn uuid_v7() -> Result<String, UuidV7Error> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| UuidV7Error::InvalidTimestamp)?
        .as_millis()
        .try_into()
        .map_err(|_| UuidV7Error::InvalidTimestamp)?;
    generate(timestamp, true)
}

pub fn uuid_v7_at(timestamp: u64) -> Result<String, UuidV7Error> {
    generate(timestamp, false)
}

fn generate(timestamp: u64, ordinary: bool) -> Result<String, UuidV7Error> {
    if timestamp > MAX_TIMESTAMP {
        return Err(UuidV7Error::InvalidTimestamp);
    }
    let mut state = state().lock().expect("UUIDv7 state lock");
    let timestamp = if ordinary {
        let timestamp = timestamp.max(state.last_ordinary_timestamp);
        state.last_ordinary_timestamp = timestamp;
        timestamp
    } else {
        timestamp
    };
    let mut bytes = rand::random::<[u8; 16]>();
    let sequence = match state.sequence {
        Some(MAX_SEQUENCE) => return Err(UuidV7Error::SequenceExhausted),
        Some(sequence) => sequence + 1,
        None => u64::from_be_bytes([0, 0, 0, bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]]),
    };
    state.sequence = Some(sequence);
    drop(state);
    let timestamp = timestamp.to_be_bytes();
    bytes[..6].copy_from_slice(&timestamp[2..]);
    bytes[6] = 0x70 | ((sequence >> 37) & 0x0f) as u8;
    bytes[7] = ((sequence >> 29) & 0xff) as u8;
    bytes[8] = 0x80 | ((sequence >> 23) & 0x3f) as u8;
    bytes[9] = ((sequence >> 15) & 0xff) as u8;
    bytes[10] = ((sequence >> 7) & 0xff) as u8;
    bytes[11] = (((sequence & 0x7f) << 1) as u8) | (bytes[11] & 0x01);
    Ok(format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        u32::from_be_bytes(bytes[..4].try_into().expect("four UUID bytes")),
        u16::from_be_bytes(bytes[4..6].try_into().expect("two UUID bytes")),
        u16::from_be_bytes(bytes[6..8].try_into().expect("two UUID bytes")),
        u16::from_be_bytes(bytes[8..10].try_into().expect("two UUID bytes")),
        u64::from_be_bytes([
            0, 0, bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        ])
    ))
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}
