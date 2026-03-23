use clap::{Parser, Subcommand};
use qemb_service::{Server, ServerConfig};
use std::net::SocketAddr;

#[derive(Parser)]
#[command(name = "qemb")]
#[command(about = "Qwen3 Embedding Bare-Metal RDNA3 Service", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the embedding service
    Serve {
        /// Path to model bundle
        #[arg(short, long, default_value = "model")]
        model: String,
        /// Port to listen on
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
    /// Convert Hugging Face model to packed format
    Convert {
        /// Path to Hugging Face model directory
        #[arg(short, long)]
        input: String,
        /// Path to output packed model
        #[arg(short, long)]
        output: String,
    },
    /// Run offline embedding inference
    Run {
        /// Input text to embed
        #[arg(short, long)]
        text: String,
        /// Path to model bundle
        #[arg(short, long, default_value = "model")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { model, port } => {
            let config = ServerConfig {
                model_name: "Qwen3-Embedding-0.6B".to_string(),
                model_path: model,
            };

            let server = Server::new(config);
            let addr: SocketAddr = ([0, 0, 0, 0], port).into();

            tracing::info!("Starting server on {}", addr);
            server.run(addr).await?;
        }
        Commands::Convert { input, output } => {
            println!("Converting {} to {}", input, output);
            println!("ERROR: Converter not yet implemented");
            std::process::exit(1);
        }
        Commands::Run { text, model } => {
            println!("Running inference on '{}' with model {}", text, model);
            println!("ERROR: Offline runner not yet implemented");
            std::process::exit(1);
        }
    }

    Ok(())
}