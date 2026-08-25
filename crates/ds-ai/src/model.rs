use crate::Usage;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{collections::BTreeMap, fmt, marker::PhantomData, ops::Deref};
use thiserror::Error;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Api {
    OpenAiResponses,
    AnthropicMessages,
    OpenAiCodexResponses,
    Other(String),
}

impl Api {
    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenAiResponses => "openai-responses",
            Self::AnthropicMessages => "anthropic-messages",
            Self::OpenAiCodexResponses => "openai-codex-responses",
            Self::Other(value) => value,
        }
    }
}

impl From<&str> for Api {
    fn from(value: &str) -> Self {
        match value {
            "openai-responses" => Self::OpenAiResponses,
            "anthropic-messages" => Self::AnthropicMessages,
            "openai-codex-responses" => Self::OpenAiCodexResponses,
            value => Self::Other(value.into()),
        }
    }
}

impl From<String> for Api {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl fmt::Display for Api {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for Api {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Api {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self::from)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderId(String);

impl ProviderId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ProviderId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ProviderId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

const THINKING_LEVELS: [ThinkingLevel; 7] = [
    ThinkingLevel::Off,
    ThinkingLevel::Minimal,
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
    ThinkingLevel::XHigh,
    ThinkingLevel::Max,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelInput {
    Text,
    Image,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    #[serde(flatten)]
    pub rates: ModelCostRates,
    pub input_tokens_above: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    #[serde(flatten)]
    pub rates: ModelCostRates,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiResponsesCompatibility {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    #[serde(
        rename = "supportsOpenAIGrammarTools",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_open_ai_grammar_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_additional_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_explicit_prompt_cache_mode: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicMessagesCompatibility {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_temperature: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_empty_signature: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_fallback_models: Vec<AnthropicFallbackModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_references: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SessionAffinityFormat {
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "openai-nosession")]
    OpenAiNoSession,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicFallbackModel {
    pub provider: ProviderId,
    pub model: String,
    pub cost: ModelCost,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelCompatibility {
    OpenAi(OpenAiResponsesCompatibility),
    Anthropic(AnthropicMessagesCompatibility),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: ProviderId,
    pub base_url: String,
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub thinking_level_map: BTreeMap<ThinkingLevel, Option<String>>,
    pub input: Vec<ModelInput>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sampling_params: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompatibility>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApiModel<O: crate::ApiOptions> {
    model: Model,
    options: PhantomData<fn() -> O>,
}

impl<O: crate::ApiOptions> ApiModel<O> {
    pub fn new(model: Model) -> Result<Self, ApiModelError> {
        let expected = O::api();
        if model.api != expected {
            return Err(ApiModelError {
                actual: model.api,
                expected,
            });
        }
        Ok(Self {
            model,
            options: PhantomData,
        })
    }

    pub fn as_model(&self) -> &Model {
        &self.model
    }

    pub fn into_model(self) -> Model {
        self.model
    }
}

impl<O: crate::ApiOptions> Deref for ApiModel<O> {
    type Target = Model;

    fn deref(&self) -> &Self::Target {
        self.as_model()
    }
}

impl<O: crate::ApiOptions> AsRef<Model> for ApiModel<O> {
    fn as_ref(&self) -> &Model {
        self.as_model()
    }
}

impl<O: crate::ApiOptions> Serialize for ApiModel<O> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.model.serialize(serializer)
    }
}

impl<'de, O: crate::ApiOptions> Deserialize<'de> for ApiModel<O> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(Model::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("model API {actual} does not match options API {expected}")]
pub struct ApiModelError {
    pub actual: Api,
    pub expected: Api,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelWire {
    id: String,
    name: String,
    api: Api,
    provider: ProviderId,
    base_url: String,
    reasoning: bool,
    #[serde(default)]
    thinking_level_map: BTreeMap<ThinkingLevel, Option<String>>,
    input: Vec<ModelInput>,
    cost: ModelCost,
    context_window: u64,
    max_tokens: u64,
    #[serde(default)]
    sampling_params: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    compat: Option<serde_json::Value>,
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModelWire::deserialize(deserializer)?;
        let compat = match (&wire.api, wire.compat) {
            (_, None) => None,
            (Api::AnthropicMessages, Some(value)) => Some(ModelCompatibility::Anthropic(
                serde_json::from_value(value).map_err(serde::de::Error::custom)?,
            )),
            (_, Some(value)) => Some(ModelCompatibility::OpenAi(
                serde_json::from_value(value).map_err(serde::de::Error::custom)?,
            )),
        };
        Ok(Self {
            id: wire.id,
            name: wire.name,
            api: wire.api,
            provider: wire.provider,
            base_url: wire.base_url,
            reasoning: wire.reasoning,
            thinking_level_map: wire.thinking_level_map,
            input: wire.input,
            cost: wire.cost,
            context_window: wire.context_window,
            max_tokens: wire.max_tokens,
            sampling_params: wire.sampling_params,
            headers: wire.headers,
            compat,
        })
    }
}

impl Model {
    pub fn typed<O: crate::ApiOptions>(&self) -> Result<ApiModel<O>, ApiModelError> {
        ApiModel::new(self.clone())
    }

    pub fn supported_thinking_levels(&self) -> Vec<ThinkingLevel> {
        if !self.reasoning {
            return vec![ThinkingLevel::Off];
        }
        THINKING_LEVELS
            .into_iter()
            .filter(|level| {
                let mapped = self.thinking_level_map.get(level);
                if mapped == Some(&None) {
                    return false;
                }
                !matches!(level, ThinkingLevel::XHigh | ThinkingLevel::Max) || mapped.is_some()
            })
            .collect()
    }

    pub fn clamp_thinking_level(&self, requested: ThinkingLevel) -> ThinkingLevel {
        let supported = self.supported_thinking_levels();
        if supported.contains(&requested) {
            return requested;
        }
        let requested_index = THINKING_LEVELS
            .iter()
            .position(|level| *level == requested)
            .unwrap_or_default();
        THINKING_LEVELS[requested_index..]
            .iter()
            .chain(THINKING_LEVELS[..requested_index].iter().rev())
            .find(|level| supported.contains(level))
            .copied()
            .unwrap_or(ThinkingLevel::Off)
    }

    pub fn calculate_cost<'a>(&self, usage: &'a mut Usage) -> &'a crate::UsageCost {
        let input_tokens = usage.input + usage.cache_read + usage.cache_write;
        let rates = self
            .cost
            .tiers
            .iter()
            .filter(|tier| input_tokens > tier.input_tokens_above)
            .max_by_key(|tier| tier.input_tokens_above)
            .map_or(&self.cost.rates, |tier| &tier.rates);
        let long_write = usage.cache_write_1h.unwrap_or_default();
        let short_write = usage.cache_write.saturating_sub(long_write);
        usage.cost.input = rates.input * usage.input as f64 / 1_000_000.0;
        usage.cost.output = rates.output * usage.output as f64 / 1_000_000.0;
        usage.cost.cache_read = rates.cache_read * usage.cache_read as f64 / 1_000_000.0;
        usage.cost.cache_write = (rates.cache_write * short_write as f64
            + rates.input * 2.0 * long_write as f64)
            / 1_000_000.0;
        usage.cost.total =
            usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
        &usage.cost
    }

    pub fn is_same_as(&self, other: &Self) -> bool {
        self.id == other.id && self.provider == other.provider
    }
}
