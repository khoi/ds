use crate::{Error, TimeoutPhase, sse};
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::Response;
use std::{
    future::{Future, pending},
    pin::Pin,
    time::Duration,
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

pub(crate) async fn connect<F>(
    request: F,
    connection_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
) -> Result<Response, Error>
where
    F: FutureResponse,
{
    tokio::pin!(request);
    tokio::select! {
        biased;
        response = &mut request => response,
        _ = wait_for(connection_timeout) => Err(Error::Timeout {
            phase: TimeoutPhase::Connection,
            partial: None,
        }),
        _ = wait_until(overall_deadline) => Err(Error::Timeout {
            phase: TimeoutPhase::Overall,
            partial: None,
        }),
    }
}

pub(crate) trait FutureResponse: Future<Output = Result<Response, Error>> + Send {}

impl<T> FutureResponse for T where T: Future<Output = Result<Response, Error>> + Send {}

pub(crate) struct EventStream {
    chunks: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    decoder: sse::Decoder,
    cancellation: CancellationToken,
    idle_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
    event_deadline: Option<Instant>,
    saw_event: bool,
}

impl EventStream {
    pub(crate) fn new(
        response: Response,
        cancellation: CancellationToken,
        first_event_timeout: Option<Duration>,
        idle_timeout: Option<Duration>,
        overall_deadline: Option<Instant>,
    ) -> Self {
        Self {
            chunks: Box::pin(response.bytes_stream()),
            decoder: sse::Decoder::default(),
            cancellation,
            idle_timeout,
            overall_deadline,
            event_deadline: first_event_timeout.map(|timeout| Instant::now() + timeout),
            saw_event: false,
        }
    }

    pub(crate) async fn next(&mut self) -> Result<Option<String>, ReadError> {
        loop {
            match self.decoder.next_data() {
                Ok(Some(data)) => {
                    self.saw_event = true;
                    self.event_deadline = self.idle_timeout.map(|timeout| Instant::now() + timeout);
                    return Ok(Some(data));
                }
                Ok(None) => {}
                Err(error) => return Err(ReadError::Stream(error)),
            }

            let chunk = tokio::select! {
                biased;
                _ = self.cancellation.cancelled() => return Err(ReadError::Cancelled),
                _ = wait_until(self.overall_deadline) => {
                    return Err(ReadError::Timeout(TimeoutPhase::Overall));
                }
                _ = wait_until(self.event_deadline) => {
                    return Err(ReadError::Timeout(if self.saw_event {
                        TimeoutPhase::Idle
                    } else {
                        TimeoutPhase::FirstEvent
                    }));
                }
                chunk = self.chunks.next() => chunk,
            };
            let Some(chunk) = chunk else {
                return Ok(None);
            };
            self.decoder
                .push(&chunk.map_err(|error| ReadError::Stream(error.to_string()))?);
        }
    }
}

pub(crate) enum ReadError {
    Cancelled,
    Timeout(TimeoutPhase),
    Stream(String),
}

async fn wait_for(timeout: Option<Duration>) {
    match timeout {
        Some(timeout) => tokio::time::sleep(timeout).await,
        None => pending().await,
    }
}

async fn wait_until(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => pending().await,
    }
}
