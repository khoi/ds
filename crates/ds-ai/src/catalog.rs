use crate::{Api, Model, ModelCompatibility, ModelInput, Provider, ProviderId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc, sync::OnceLock};
use thiserror::Error;

const OPENAI_JSON: &str = include_str!("catalog/openai.json");
const ANTHROPIC_JSON: &str = include_str!("catalog/anthropic.json");
const CODEX_JSON: &str = include_str!("catalog/openai-codex.json");
const MANIFEST_JSON: &str = include_str!("catalog/manifest.json");

static OPENAI: OnceLock<Vec<Model>> = OnceLock::new();
static ANTHROPIC: OnceLock<Vec<Model>> = OnceLock::new();
static CODEX: OnceLock<Vec<Model>> = OnceLock::new();
static MANIFEST: OnceLock<CatalogInfo> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogInfo {
    pub schema_version: u64,
    pub source_commit: String,
    pub generated_at: String,
    pub files: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{message}")]
pub struct CatalogError {
    message: String,
}

pub fn builtin_catalog_info() -> &'static CatalogInfo {
    MANIFEST.get_or_init(|| {
        serde_json::from_str(MANIFEST_JSON).expect("valid built-in catalog manifest")
    })
}

pub fn openai_models() -> &'static [Model] {
    OPENAI.get_or_init(|| load(OPENAI_JSON, Api::OpenAiResponses, "openai", "openai.json"))
}

pub fn anthropic_models() -> &'static [Model] {
    ANTHROPIC.get_or_init(|| {
        load(
            ANTHROPIC_JSON,
            Api::AnthropicMessages,
            "anthropic",
            "anthropic.json",
        )
    })
}

pub fn codex_models() -> &'static [Model] {
    CODEX.get_or_init(|| {
        load(
            CODEX_JSON,
            Api::OpenAiCodexResponses,
            "openai-codex",
            "openai-codex.json",
        )
    })
}

pub fn builtin_model(provider: &str, id: &str) -> Option<Model> {
    builtin_provider_models(provider)
        .iter()
        .find(|model| model.id == id)
        .cloned()
}

pub fn builtin_provider_models(provider: &str) -> &'static [Model] {
    match provider {
        "openai" => openai_models(),
        "anthropic" => anthropic_models(),
        "openai-codex" => codex_models(),
        _ => &[],
    }
}

pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    vec![
        crate::openai::provider(),
        crate::anthropic::provider(),
        crate::codex::provider(),
    ]
}

pub fn builtin_models() -> crate::Models {
    let mut models = crate::Models::new();
    for provider in builtin_providers() {
        models.set_provider(provider);
    }
    models
}

pub fn validate_builtin_catalog() -> Result<(), CatalogError> {
    validate_file("openai.json", OPENAI_JSON)?;
    validate_file("anthropic.json", ANTHROPIC_JSON)?;
    validate_file("openai-codex.json", CODEX_JSON)?;
    validate_model_catalog(openai_models(), &Api::OpenAiResponses, "openai")?;
    validate_model_catalog(anthropic_models(), &Api::AnthropicMessages, "anthropic")?;
    validate_model_catalog(codex_models(), &Api::OpenAiCodexResponses, "openai-codex")
}

pub fn validate_model_catalog(
    models: &[Model],
    api: &Api,
    provider: &str,
) -> Result<(), CatalogError> {
    let mut ids = std::collections::BTreeSet::new();
    for model in models {
        validate_model(model, api, provider)?;
        if !ids.insert(&model.id) {
            return fail(format!("duplicate model {provider}/{}", model.id));
        }
    }
    Ok(())
}

fn load(source: &str, api: Api, provider: &str, file: &str) -> Vec<Model> {
    validate_file(file, source).unwrap_or_else(|error| panic!("{error}"));
    let mut catalog = serde_json::from_str::<BTreeMap<String, BTreeMap<String, Model>>>(source)
        .unwrap_or_else(|error| panic!("invalid built-in catalog {file}: {error}"));
    let models = catalog
        .remove(api.as_str())
        .unwrap_or_else(|| panic!("built-in catalog {file} has no {} models", api));
    assert!(catalog.is_empty(), "built-in catalog {file} has extra APIs");
    for (id, model) in &models {
        assert_eq!(
            id, &model.id,
            "built-in catalog {file} has a mismatched key"
        );
    }
    let models = models.into_values().collect::<Vec<_>>();
    validate_model_catalog(&models, &api, provider).unwrap_or_else(|error| panic!("{error}"));
    models
}

fn validate_file(file: &str, source: &str) -> Result<(), CatalogError> {
    let expected = builtin_catalog_info()
        .files
        .get(file)
        .ok_or_else(|| error(format!("catalog manifest has no hash for {file}")))?;
    let actual = format!("{:x}", Sha256::digest(source.as_bytes()));
    if &actual != expected {
        return fail(format!("catalog hash mismatch for {file}"));
    }
    Ok(())
}

fn validate_model(model: &Model, api: &Api, provider: &str) -> Result<(), CatalogError> {
    let name = format!("{provider}/{}", model.id);
    if model.id.is_empty() || model.name.is_empty() || model.base_url.is_empty() {
        return fail(format!("model {name} has empty identity data"));
    }
    if &model.api != api || model.provider != ProviderId::new(provider) {
        return fail(format!("model {name} has the wrong API or provider"));
    }
    if model.context_window == 0 || model.max_tokens == 0 || model.max_tokens > model.context_window
    {
        return fail(format!("model {name} has invalid token limits"));
    }
    if model.input.is_empty()
        || !model.input.contains(&ModelInput::Text)
        || model
            .input
            .iter()
            .filter(|input| **input == ModelInput::Text)
            .count()
            > 1
        || model
            .input
            .iter()
            .filter(|input| **input == ModelInput::Image)
            .count()
            > 1
    {
        return fail(format!("model {name} has invalid input capabilities"));
    }
    validate_rates(&name, &model.cost.rates)?;
    let mut thresholds = std::collections::BTreeSet::new();
    for tier in &model.cost.tiers {
        if tier.input_tokens_above == 0 || !thresholds.insert(tier.input_tokens_above) {
            return fail(format!("model {name} has invalid cost tiers"));
        }
        validate_rates(&name, &tier.rates)?;
    }
    if matches!(
        (&model.compat, api),
        (
            Some(ModelCompatibility::Anthropic(_)),
            Api::OpenAiResponses | Api::OpenAiCodexResponses
        ) | (Some(ModelCompatibility::OpenAi(_)), Api::AnthropicMessages)
    ) {
        return fail(format!("model {name} has compatibility for the wrong API"));
    }
    Ok(())
}

fn validate_rates(name: &str, rates: &crate::ModelCostRates) -> Result<(), CatalogError> {
    if [
        rates.input,
        rates.output,
        rates.cache_read,
        rates.cache_write,
    ]
    .into_iter()
    .any(|rate| !rate.is_finite() || rate < 0.0)
    {
        return fail(format!("model {name} has invalid costs"));
    }
    Ok(())
}

fn fail<T>(message: String) -> Result<T, CatalogError> {
    Err(error(message))
}

fn error(message: String) -> CatalogError {
    CatalogError { message }
}
