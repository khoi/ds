mod http;
mod json;
mod retry;
mod schema;
mod sse;
mod transport;
mod types;

pub mod anthropic;
pub mod codex;
pub mod openai;

pub use types::{
    CacheRetention, Content, Context, Error, Event, InputContent, Message, RateLimits, Response,
    ResponseMetadata, ResponseStream, StopReason, TimeoutPhase, Tool, ToolCall, ToolResult, Usage,
};

pub async fn complete(mut stream: ResponseStream) -> Result<Response, Error> {
    use futures_util::StreamExt;

    while let Some(event) = stream.next().await {
        if let Event::Done(response) = event? {
            return Ok(*response);
        }
    }
    Err(Error::IncompleteStream {
        partial: Response::default(),
    })
}
