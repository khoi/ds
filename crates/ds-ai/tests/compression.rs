use base64::prelude::*;
use brotli::CompressorWriter;
use ds_ai::{
    AnthropicOptions, Context, Credential, Message, OAuthAuth, OpenAiCodexResponsesOptions,
    OpenAiResponsesOptions, StreamOptions, Transport, anthropic, builtin_model, codex, openai,
};
use flate2::{
    Compression,
    write::{GzEncoder, ZlibEncoder},
};
use serde_json::json;
use std::{collections::BTreeMap, io::Write};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

const ENCODINGS: [&str; 3] = ["gzip", "deflate", "br"];

#[tokio::test]
async fn openai_streams_decode_standard_http_compression() {
    let body = b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"compressed-openai\",\"status\":\"completed\",\"usage\":{}}}\n\n";

    for encoding in ENCODINGS {
        let server = Server::new("text/event-stream", encoding, body).await;
        let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
        model.base_url = server.base_url.clone();
        let options = OpenAiResponsesOptions {
            stream: authenticated_stream_options("key"),
            ..Default::default()
        };

        let result = openai::stream(
            &model.typed::<OpenAiResponsesOptions>().unwrap(),
            &Context::new([Message::user("Hello")]),
            &options,
        )
        .result()
        .await
        .unwrap();

        assert_eq!(result.response_id.as_deref(), Some("compressed-openai"));
        server.finish().await;
    }
}

#[tokio::test]
async fn anthropic_streams_decode_standard_http_compression() {
    let body = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"compressed-anthropic\",\"usage\":{}}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();

    for encoding in ENCODINGS {
        let server = Server::new("text/event-stream", encoding, body.as_bytes()).await;
        let mut model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
        model.base_url = server.base_url.clone();
        let options = AnthropicOptions {
            stream: authenticated_stream_options("key"),
            ..Default::default()
        };

        let result = anthropic::stream(
            &model.typed::<AnthropicOptions>().unwrap(),
            &Context::new([Message::user("Hello")]),
            &options,
        )
        .result()
        .await
        .unwrap();

        assert_eq!(result.response_id.as_deref(), Some("compressed-anthropic"));
        server.finish().await;
    }
}

#[tokio::test]
async fn codex_sse_streams_decode_standard_http_compression() {
    let body = b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"compressed-codex\",\"status\":\"completed\",\"usage\":{}}}\n\n";

    for encoding in ENCODINGS {
        let server = Server::new("text/event-stream", encoding, body).await;
        let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
        model.base_url = server.base_url.clone();
        let options = OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                transport: Some(Transport::Sse),
                ..authenticated_stream_options(&codex_token("account"))
            },
            ..Default::default()
        };

        let result = codex::stream(
            &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
            &Context::new([Message::user("Hello")]),
            &options,
        )
        .result()
        .await
        .unwrap();

        assert_eq!(result.response_id.as_deref(), Some("compressed-codex"));
        server.finish().await;
    }
}

#[tokio::test]
async fn anthropic_oauth_decodes_standard_http_compression() {
    let body = serde_json::to_vec(&json!({
        "access_token": "anthropic-access",
        "refresh_token": "anthropic-refresh",
        "expires_in": 3600
    }))
    .unwrap();

    for encoding in ENCODINGS {
        let server = Server::new("application/json", encoding, &body).await;
        let oauth = anthropic::auth::OAuth::new()
            .with_token_url(format!("{}/oauth/token", server.base_url));
        let credential = Credential::OAuth {
            refresh: "old-refresh".into(),
            access: "old-access".into(),
            expires: 0,
            extra: BTreeMap::new(),
        };

        let result = oauth
            .refresh(&credential, &CancellationToken::new())
            .await
            .unwrap();

        assert!(matches!(
            result,
            Credential::OAuth { access, refresh, .. }
                if access == "anthropic-access" && refresh == "anthropic-refresh"
        ));
        server.finish().await;
    }
}

#[tokio::test]
async fn codex_oauth_decodes_standard_http_compression() {
    for encoding in ENCODINGS {
        let access = codex_token("refreshed-account");
        let body = serde_json::to_vec(&json!({
            "access_token": access,
            "refresh_token": "codex-refresh",
            "expires_in": 3600
        }))
        .unwrap();
        let server = Server::new("application/json", encoding, &body).await;
        let oauth = codex::auth::OAuth::new().with_base_url(&server.base_url);
        let credential = Credential::OAuth {
            refresh: "old-refresh".into(),
            access: codex_token("old-account"),
            expires: 0,
            extra: BTreeMap::new(),
        };

        let result = oauth
            .refresh(&credential, &CancellationToken::new())
            .await
            .unwrap();

        assert!(matches!(
            result,
            Credential::OAuth { access, refresh, .. }
                if access == codex_token("refreshed-account") && refresh == "codex-refresh"
        ));
        server.finish().await;
    }
}

fn authenticated_stream_options(api_key: &str) -> StreamOptions {
    StreamOptions {
        api_key: Some(api_key.into()),
        ..Default::default()
    }
}

fn codex_token(account_id: &str) -> String {
    let payload = BASE64_URL_SAFE_NO_PAD.encode(
        serde_json::to_vec(&json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id}
        }))
        .unwrap(),
    );
    format!("header.{payload}.signature")
}

struct Server {
    base_url: String,
    task: JoinHandle<Vec<u8>>,
}

impl Server {
    async fn new(content_type: &'static str, encoding: &'static str, body: &[u8]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = compress(encoding, body);
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_request(&mut socket).await;
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-encoding: {encoding}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            socket.write_all(&body).await.unwrap();
            request
        });
        Self {
            base_url: format!("http://{address}"),
            task,
        }
    }

    async fn finish(self) {
        assert!(!self.task.await.unwrap().is_empty());
    }
}

async fn read_request(socket: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let header_end = loop {
        let mut bytes = [0; 1024];
        let count = socket.read(&mut bytes).await.unwrap();
        request.extend_from_slice(&bytes[..count]);
        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or_default();
    while request.len() < header_end + content_length {
        let mut bytes = [0; 1024];
        let count = socket.read(&mut bytes).await.unwrap();
        request.extend_from_slice(&bytes[..count]);
    }
    request
}

fn compress(encoding: &str, input: &[u8]) -> Vec<u8> {
    match encoding {
        "gzip" => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(input).unwrap();
            encoder.finish().unwrap()
        }
        "deflate" => {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(input).unwrap();
            encoder.finish().unwrap()
        }
        "br" => {
            let mut output = Vec::new();
            CompressorWriter::new(&mut output, 4096, 5, 22)
                .write_all(input)
                .unwrap();
            output
        }
        _ => unreachable!(),
    }
}
