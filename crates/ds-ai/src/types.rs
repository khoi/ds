use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Message {
    User(String),
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self::User(content.into())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Context {
    messages: Vec<Message>,
}

impl Context {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
        }
    }

    pub(crate) fn messages(&self) -> &[Message] {
        &self.messages
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Content {
    Text(String),
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub reasoning: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub id: Option<String>,
    pub content: Vec<Content>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    TextDelta { content_index: usize, delta: String },
    Done(Response),
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(String),
    #[error("provider returned HTTP {status}: {body}")]
    Provider { status: u16, body: String },
    #[error("invalid provider stream: {0}")]
    Stream(String),
}

pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
