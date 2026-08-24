mod event_stream;
mod frame;
mod http;
mod json;
mod legacy;
mod message;
mod model;
mod provider;
mod retry;
mod schema;
mod sse;
mod transport;
mod types;

pub mod anthropic;
pub mod codex;
pub mod openai;

pub use event_stream::{AssistantMessageEventStream, AssistantMessageStreamError};
pub use frame::{
    AssistantMessageFrame, AssistantMessageFrameError, assistant_message_event_to_frame,
    reduce_assistant_message_frames,
};
pub use message::{
    AssistantContent, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    AssistantToolCall, DiagnosticError, ImageContent, TextContent, ThinkingContent,
};
pub use model::{
    AnthropicFallbackModel, AnthropicMessagesCompatibility, Api, Model, ModelCompatibility,
    ModelCost, ModelCostRates, ModelCostTier, ModelInput, OpenAiResponsesCompatibility, ProviderId,
    SessionAffinityFormat, ThinkingLevel,
};
pub use provider::{Models, Provider, SimpleStreamOptions, StreamOptions, ToolChoice, Transport};
pub use types::{
    CacheRetention, ConstrainedSampling, ConstrainedSamplingStrictness, Content, Context, Error,
    Event, GrammarVariants, InputContent, Message, RateLimits, Response, ResponseMetadata,
    ResponseStream, StopReason, TimeoutPhase, Tool, ToolCall, ToolResultMessage, ToolResultRole,
    Usage, UsageCost, UserContent, UserMessage, UserRole,
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
