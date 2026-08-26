use async_trait::async_trait;
use ds_ai::Tool;
use std::{collections::HashMap, path::Path, sync::Arc};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub struct AgentTool {
    declaration: Tool,
    executor: Arc<dyn ToolExecutor>,
}

impl AgentTool {
    pub fn new(declaration: Tool, executor: impl ToolExecutor + 'static) -> Self {
        Self {
            declaration,
            executor: Arc::new(executor),
        }
    }

    pub fn declaration(&self) -> &Tool {
        &self.declaration
    }

    pub(crate) async fn execute(
        &self,
        arguments: serde_json::Value,
        context: ToolExecutionContext<'_>,
    ) -> ToolOutput {
        self.executor.execute(arguments, context).await
    }
}

pub struct ToolRegistry {
    tools: Vec<AgentTool>,
    by_name: HashMap<String, usize>,
}

impl ToolRegistry {
    pub fn new(tools: impl IntoIterator<Item = AgentTool>) -> Result<Self, DuplicateToolError> {
        let tools = tools.into_iter().collect::<Vec<_>>();
        let mut by_name = HashMap::with_capacity(tools.len());
        for (index, tool) in tools.iter().enumerate() {
            let name = tool.declaration.name.clone();
            if by_name.insert(name.clone(), index).is_some() {
                return Err(DuplicateToolError(name));
            }
        }
        Ok(Self { tools, by_name })
    }

    pub fn declarations(&self) -> Vec<Tool> {
        self.tools
            .iter()
            .map(|tool| tool.declaration.clone())
            .collect()
    }

    pub(crate) fn get(&self, name: &str) -> Option<&AgentTool> {
        self.by_name.get(name).map(|index| &self.tools[*index])
    }
}

#[derive(Debug, Error, PartialEq)]
#[error("duplicate tool name: {0}")]
pub struct DuplicateToolError(pub String);

pub struct ToolExecutionContext<'a> {
    pub working_directory: &'a Path,
    pub cancellation: &'a CancellationToken,
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(
        &self,
        arguments: serde_json::Value,
        context: ToolExecutionContext<'_>,
    ) -> ToolOutput;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundedText {
    pub text: String,
    pub truncated: bool,
}

impl BoundedText {
    pub fn new(text: impl Into<String>, truncated: bool) -> Self {
        Self {
            text: text.into(),
            truncated,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolOutput {
    pub content: BoundedText,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn success(content: BoundedText) -> Self {
        Self {
            content,
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: BoundedText::new(content, false),
            is_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Noop;

    #[async_trait]
    impl ToolExecutor for Noop {
        async fn execute(
            &self,
            _arguments: serde_json::Value,
            _context: ToolExecutionContext<'_>,
        ) -> ToolOutput {
            ToolOutput::success(BoundedText::new("", false))
        }
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let declaration = || Tool::new("read", "read", json!({ "type": "object" }));
        let error = ToolRegistry::new([
            AgentTool::new(declaration(), Noop),
            AgentTool::new(declaration(), Noop),
        ])
        .err()
        .expect("duplicate rejected");

        assert_eq!(error, DuplicateToolError("read".into()));
    }
}
