use crate::{AssistantMessage, AssistantMessageEvent};
use futures_core::Stream;
use futures_util::StreamExt;
use std::{
    pin::Pin,
    task::{Context, Poll},
};
use thiserror::Error;

pub struct AssistantMessageEventStream {
    inner: Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send>>,
    terminal: Option<AssistantMessage>,
    settled: bool,
}

impl AssistantMessageEventStream {
    pub fn new(stream: impl Stream<Item = AssistantMessageEvent> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(stream),
            terminal: None,
            settled: false,
        }
    }

    pub async fn result(&mut self) -> Result<AssistantMessage, AssistantMessageStreamError> {
        if let Some(message) = &self.terminal {
            return Ok(message.clone());
        }
        while self.next().await.is_some() {}
        self.terminal
            .clone()
            .ok_or(AssistantMessageStreamError::MissingTerminalEvent)
    }
}

impl Stream for AssistantMessageEventStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.settled {
            return Poll::Ready(None);
        }
        match self.inner.as_mut().poll_next(context) {
            Poll::Ready(Some(event)) => {
                if let Some(message) = terminal(&event) {
                    self.terminal = Some(message.clone());
                    self.settled = true;
                }
                Poll::Ready(Some(event))
            }
            Poll::Ready(None) => {
                self.settled = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn terminal(event: &AssistantMessageEvent) -> Option<&AssistantMessage> {
    match event {
        AssistantMessageEvent::Done { message, .. } => Some(message),
        AssistantMessageEvent::Error { error, .. } => Some(error),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum AssistantMessageStreamError {
    #[error("assistant message stream ended without a terminal event")]
    MissingTerminalEvent,
}
