use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{Notify, oneshot},
};

pub struct Reply {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    headers: Vec<(&'static str, String)>,
    chunks: Vec<Vec<u8>>,
    disconnect: bool,
    wait_before_headers: bool,
    finish: bool,
}

impl Reply {
    pub fn sse(body: impl Into<Vec<u8>>) -> Self {
        Self::sse_chunks([body.into()])
    }

    pub fn sse_chunks(chunks: impl IntoIterator<Item = Vec<u8>>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            content_type: "text/event-stream",
            headers: Vec::new(),
            chunks: chunks.into_iter().collect(),
            disconnect: false,
            wait_before_headers: false,
            finish: true,
        }
    }

    pub fn json(status: u16, body: Value) -> Self {
        Self {
            status,
            reason: match status {
                429 => "Too Many Requests",
                500 => "Internal Server Error",
                502 => "Bad Gateway",
                503 => "Service Unavailable",
                _ => "Error",
            },
            content_type: "application/json",
            headers: Vec::new(),
            chunks: vec![serde_json::to_vec(&body).unwrap()],
            disconnect: false,
            wait_before_headers: false,
            finish: true,
        }
    }

    pub fn disconnect() -> Self {
        Self {
            status: 0,
            reason: "",
            content_type: "",
            headers: Vec::new(),
            chunks: Vec::new(),
            disconnect: true,
            wait_before_headers: false,
            finish: true,
        }
    }

    pub fn pending() -> Self {
        Self {
            status: 0,
            reason: "",
            content_type: "",
            headers: Vec::new(),
            chunks: Vec::new(),
            disconnect: false,
            wait_before_headers: true,
            finish: false,
        }
    }

    pub fn open_sse(body: impl Into<Vec<u8>>) -> Self {
        Self {
            finish: false,
            ..Self::sse(body)
        }
    }

    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }
}

pub struct Server {
    pub base_url: String,
    requests: oneshot::Receiver<Vec<String>>,
    request_count: Arc<AtomicUsize>,
    request_notify: Arc<Notify>,
}

impl Server {
    pub fn request_count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }

    pub async fn wait_for_requests(&self, count: usize) {
        loop {
            let notified = self.request_notify.notified();
            if self.request_count() >= count {
                return;
            }
            notified.await;
        }
    }

    pub async fn requests(self) -> Vec<String> {
        self.requests.await.unwrap()
    }
}

pub async fn serve(replies: impl IntoIterator<Item = Reply>) -> Server {
    let replies = replies.into_iter().collect::<Vec<_>>();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (request_sender, request_receiver) = oneshot::channel();
    let request_count = Arc::new(AtomicUsize::new(0));
    let request_notify = Arc::new(Notify::new());
    let task_request_count = request_count.clone();
    let task_request_notify = request_notify.clone();

    tokio::spawn(async move {
        let mut requests = Vec::with_capacity(replies.len());
        for reply in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            requests.push(read_request(&mut socket).await);
            task_request_count.fetch_add(1, Ordering::SeqCst);
            task_request_notify.notify_waiters();
            write_reply(&mut socket, reply).await;
        }
        request_sender.send(requests).unwrap();
    });

    Server {
        base_url: format!("http://{address}"),
        requests: request_receiver,
        request_count,
        request_notify,
    }
}

async fn read_request(socket: &mut TcpStream) -> String {
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
    String::from_utf8(request).unwrap()
}

async fn write_reply(socket: &mut TcpStream, reply: Reply) {
    if reply.disconnect {
        return;
    }
    if reply.wait_before_headers {
        let mut byte = [0];
        let _ = socket.read(&mut byte).await;
        return;
    }
    let headers = reply
        .headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect::<String>();
    socket
        .write_all(
            format!(
                "HTTP/1.1 {} {}\r\ncontent-type: {}\r\ntransfer-encoding: chunked\r\nconnection: close\r\n{}\r\n",
                reply.status, reply.reason, reply.content_type, headers
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    for chunk in reply.chunks {
        socket
            .write_all(format!("{:X}\r\n", chunk.len()).as_bytes())
            .await
            .unwrap();
        socket.write_all(&chunk).await.unwrap();
        socket.write_all(b"\r\n").await.unwrap();
    }
    if !reply.finish {
        let mut byte = [0];
        let _ = socket.read(&mut byte).await;
        return;
    }
    socket.write_all(b"0\r\n\r\n").await.unwrap();
}
