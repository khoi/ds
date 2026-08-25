use crate::{Api, ProviderId, StopReason, Usage};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::SerializeStruct,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    pub data: String,
    pub mime_type: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantContent {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(AssistantToolCall),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticError {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDiagnostic {
    pub r#type: String,
    pub timestamp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<BTreeMap<String, serde_json::Value>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssistantMessage {
    pub content: Vec<AssistantContent>,
    pub api: Api,
    pub provider: ProviderId,
    pub model: String,
    pub response_model: Option<String>,
    pub response_id: Option<String>,
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    pub error_message: Option<String>,
    pub raw_stop_reason: Option<String>,
    pub end_turn: Option<bool>,
    pub timestamp: u64,
}

impl Serialize for AssistantMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("AssistantMessage", 16)?;
        state.serialize_field("role", "assistant")?;
        state.serialize_field("content", &self.content)?;
        state.serialize_field("api", &self.api)?;
        state.serialize_field("provider", &self.provider)?;
        state.serialize_field("model", &self.model)?;
        if let Some(value) = &self.response_model {
            state.serialize_field("responseModel", value)?;
        }
        if let Some(value) = &self.response_id {
            state.serialize_field("responseId", value)?;
        }
        if let Some(value) = &self.diagnostics {
            state.serialize_field("diagnostics", value)?;
        }
        state.serialize_field("usage", &self.usage)?;
        state.serialize_field("stopReason", &self.stop_reason)?;
        if let Some(value) = &self.error_message {
            state.serialize_field("errorMessage", value)?;
        }
        if let Some(value) = &self.raw_stop_reason {
            state.serialize_field("rawStopReason", value)?;
        }
        if let Some(value) = self.end_turn {
            state.serialize_field("endTurn", &value)?;
        }
        state.serialize_field("timestamp", &self.timestamp)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssistantMessageWire {
    role: String,
    content: Vec<AssistantContent>,
    api: Api,
    provider: ProviderId,
    model: String,
    response_model: Option<String>,
    response_id: Option<String>,
    diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    usage: Usage,
    stop_reason: StopReason,
    error_message: Option<String>,
    raw_stop_reason: Option<String>,
    end_turn: Option<bool>,
    timestamp: u64,
}

impl<'de> Deserialize<'de> for AssistantMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AssistantMessageWire::deserialize(deserializer)?;
        if wire.role != "assistant" {
            return Err(D::Error::custom("assistant message role must be assistant"));
        }
        Ok(Self {
            content: wire.content,
            api: wire.api,
            provider: wire.provider,
            model: wire.model,
            response_model: wire.response_model,
            response_id: wire.response_id,
            diagnostics: wire.diagnostics,
            usage: wire.usage,
            stop_reason: wire.stop_reason,
            error_message: wire.error_message,
            raw_stop_reason: wire.raw_stop_reason,
            end_turn: wire.end_turn,
            timestamp: wire.timestamp,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DoneReason {
    Stop,
    Length,
    ToolUse,
    Deferred,
}

impl TryFrom<StopReason> for DoneReason {
    type Error = StopReason;

    fn try_from(reason: StopReason) -> Result<Self, StopReason> {
        match reason {
            StopReason::Stop => Ok(Self::Stop),
            StopReason::Length => Ok(Self::Length),
            StopReason::ToolUse => Ok(Self::ToolUse),
            StopReason::Deferred => Ok(Self::Deferred),
            reason => Err(reason),
        }
    }
}

impl From<DoneReason> for StopReason {
    fn from(reason: DoneReason) -> Self {
        match reason {
            DoneReason::Stop => Self::Stop,
            DoneReason::Length => Self::Length,
            DoneReason::ToolUse => Self::ToolUse,
            DoneReason::Deferred => Self::Deferred,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorReason {
    Error,
    Aborted,
}

impl TryFrom<StopReason> for ErrorReason {
    type Error = StopReason;

    fn try_from(reason: StopReason) -> Result<Self, StopReason> {
        match reason {
            StopReason::Error => Ok(Self::Error),
            StopReason::Aborted => Ok(Self::Aborted),
            reason => Err(reason),
        }
    }
}

impl From<ErrorReason> for StopReason {
    fn from(reason: ErrorReason) -> Self {
        match reason {
            ErrorReason::Error => Self::Error,
            ErrorReason::Aborted => Self::Aborted,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ToolCallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: AssistantToolCall,
        partial: AssistantMessage,
    },
    Done {
        reason: DoneReason,
        message: AssistantMessage,
    },
    Error {
        reason: ErrorReason,
        error: AssistantMessage,
    },
}
