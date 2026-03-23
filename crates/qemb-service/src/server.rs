//! HTTP server implementation

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::net::SocketAddr;
use std::sync::Arc;

use crate::api::*;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model_name: String,
    pub model_path: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            model_name: "Qwen3-Embedding-0.6B".to_string(),
            model_path: "model".to_string(),
        }
    }
}

pub struct Server {
    config: Arc<ServerConfig>,
}

impl Server {
    pub fn new(config: ServerConfig) -> Self {
        Server {
            config: Arc::new(config),
        }
    }

    pub async fn run(self, addr: SocketAddr) -> anyhow::Result<()> {
        let app = Router::new()
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .route("/v1/models", get(list_models))
            .route("/v1/embeddings", post(create_embedding))
            .with_state(self.config);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("Server listening on {}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn healthz() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn readyz(State(_config): State<Arc<ServerConfig>>) -> impl IntoResponse {
    // TODO: Check if model and GPU are ready
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn list_models(State(config): State<Arc<ServerConfig>>) -> impl IntoResponse {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: config.model_name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "qwen".to_string(),
        }],
    })
}

async fn create_embedding(
    State(config): State<Arc<ServerConfig>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Validate model name
    if request.model != config.model_name {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Model '{}' not found", request.model),
                    error_type: "invalid_request_error".to_string(),
                    code: "model_not_found".to_string(),
                },
            }),
        ));
    }

    // Validate encoding format
    match request.encoding_format {
        EncodingFormat::Float => {}
        EncodingFormat::Base64 => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Base64 encoding not supported in v1".to_string(),
                        error_type: "invalid_request_error".to_string(),
                        code: "unsupported_encoding".to_string(),
                    },
                }),
            ));
        }
    }

    // TODO: Implement actual embedding
    // For now, return a placeholder
    let texts = match request.input {
        EmbeddingInput::Single(text) => vec![text],
        EmbeddingInput::Multiple(texts) => texts,
    };

    let data: Vec<EmbeddingData> = texts
        .iter()
        .enumerate()
        .map(|(i, _text)| {
            // Placeholder: return zeros
            EmbeddingData {
                object: "embedding".to_string(),
                embedding: vec![0.0; request.dimensions],
                index: i,
            }
        })
        .collect();

    let total_tokens: usize = texts.iter().map(|t| t.len() / 4).sum(); // Rough estimate

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: config.model_name.clone(),
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}