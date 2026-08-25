mod auth;
mod catalog;
mod constrained_sampling;
mod deferred_tools;
mod estimate;
mod event_stream;
mod frame;
mod http;
mod json;
mod message;
mod model;
mod overflow;
mod provider;
mod provider_stream;
mod retry;
mod schema;
mod sse;
mod text;
mod transport;
mod types;
mod uuid;
mod validation;

pub mod anthropic;
pub mod codex;
pub mod openai;

pub use auth::{
    ApiKeyAuth, AuthCheck, AuthContext, AuthError, AuthEvent, AuthInfoLink, AuthInteraction,
    AuthPrompt, AuthResolutionOverrides, AuthResult, AuthSelectOption, Credential, CredentialInfo,
    CredentialMutation, CredentialStore, CredentialType, EnvApiKeyAuth, InMemoryCredentialStore,
    ModelAuth, OAuthAuth, ProviderAuth, SystemAuthContext,
};
pub use catalog::{
    CatalogError, CatalogInfo, anthropic_models, builtin_anthropic_model, builtin_catalog_info,
    builtin_codex_model, builtin_model, builtin_models, builtin_openai_model,
    builtin_provider_models, builtin_providers, codex_models, openai_models,
    validate_builtin_catalog, validate_model_catalog,
};

pub use estimate::{
    ContextUsageEstimate, calculate_context_tokens, clamp_max_tokens_to_context,
    estimate_context_tokens, estimate_message_tokens, estimate_messages_tokens,
    estimate_text_tokens,
};
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
    AnthropicFallbackModel, AnthropicMessagesCompatibility, Api, ApiModel, ApiModelError, Model,
    ModelCompatibility, ModelCost, ModelCostRates, ModelCostTier, ModelInput,
    OpenAiResponsesCompatibility, ProviderId, SessionAffinityFormat, ThinkingLevel,
};
pub use overflow::{is_context_overflow, is_recoverable_length};
pub use provider::{
    AnthropicModel, AnthropicOptions, ApiOptions, ApiStreamOptions, HeaderHook, Models,
    OpenAiCodexResponsesModel, OpenAiCodexResponsesOptions, OpenAiResponsesModel,
    OpenAiResponsesOptions, PayloadHook, Provider, ProviderResponse, ResponseHook,
    SimpleStreamOptions, StreamOptions, ThinkingBudgets, ToolChoice, Transport,
};
pub use retry::{RetryCallbacks, RetryPolicy, is_retryable_assistant_error, retry_assistant_call};
pub use text::{ContentText, content_text, content_text_with_separator};
pub use types::{
    CacheRetention, ConstrainedSampling, ConstrainedSamplingStrictness, Context, GrammarVariants,
    InputContent, Message, StopReason, Tool, ToolResultMessage, Usage, UsageCost, UserContent,
    UserMessage,
};
pub(crate) use types::{Error, Response, RetryDelay, TimeoutPhase};
pub use uuid::{UuidV7Error, uuid_v7, uuid_v7_at};
pub use validation::{ToolValidationError, validate_tool_arguments, validate_tool_call};
