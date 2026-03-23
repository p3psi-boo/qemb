use clap::{Parser, Subcommand};
use qemb_convert::{BundleReader, BundleWriter, ModelBundleBuilder, ModelConfig, SafetensorsLoader};
use qemb_service::{Server, ServerConfig};
use std::net::SocketAddr;
use std::path::PathBuf;

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
        /// Model name for the bundle
        #[arg(long, default_value = "Qwen3-Embedding-0.6B")]
        name: String,
    },
    /// Inspect a packed model bundle
    Inspect {
        /// Path to model bundle
        #[arg(short, long, default_value = "model")]
        model: String,
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
        Commands::Convert { input, output, name } => {
            println!("Converting model from {} to {}", input, output);

            let input_path = PathBuf::from(&input);
            let output_path = PathBuf::from(&output);

            // Look for safetensors files
            let safetensors_files: Vec<_> = std::fs::read_dir(&input_path)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
                .map(|e| e.path())
                .collect();

            if safetensors_files.is_empty() {
                anyhow::bail!("No .safetensors files found in {:?}", input_path);
            }

            println!("Found {} safetensors file(s)", safetensors_files.len());

            // Load all tensors
            let mut all_tensors = Vec::new();
            for file in &safetensors_files {
                println!("Loading {:?}", file);
                let loader = SafetensorsLoader::from_file(file)?;
                for tensor_name in loader.names() {
                    let tensor = loader.get(tensor_name).unwrap();
                    all_tensors.push((
                        tensor_name.clone(),
                        tensor.data.clone(),
                        tensor.shape.clone(),
                    ));
                }
            }

            println!("Loaded {} tensors", all_tensors.len());

            // Build the bundle
            let config = ModelConfig::default();
            let mut builder = ModelBundleBuilder::new(&name).with_config(config);

            for (tensor_name, data, shape) in &all_tensors {
                let size_bytes = data.len();
                builder = builder.add_tensor(tensor_name, shape.clone(), "BF16", size_bytes);
            }

            let bundle = builder.build();
            let mut writer = BundleWriter::new(bundle);

            for (tensor_name, data, _shape) in &all_tensors {
                writer.write_tensor(tensor_name, data)?;
            }

            writer.write_to_dir(&output_path)?;
            println!("Model bundle written to {:?}", output_path);
        }
        Commands::Inspect { model } => {
            let model_path = PathBuf::from(&model);

            match BundleReader::read_metadata(&model_path) {
                Ok(bundle) => {
                    println!("Model: {}", bundle.name);
                    println!("Version: {}", bundle.version);
                    println!("\nConfig:");
                    println!("  hidden_size: {}", bundle.config.hidden_size);
                    println!("  intermediate_size: {}", bundle.config.intermediate_size);
                    println!("  num_layers: {}", bundle.config.num_layers);
                    println!(
                        "  num_attention_heads: {}",
                        bundle.config.num_attention_heads
                    );
                    println!(
                        "  num_key_value_heads: {}",
                        bundle.config.num_key_value_heads
                    );
                    println!("  vocab_size: {}", bundle.config.vocab_size);
                    println!(
                        "  max_position_embeddings: {}",
                        bundle.config.max_position_embeddings
                    );

                    println!(
                        "\nTensors: {} tensors, {} bytes",
                        bundle.tensors.tensors.len(),
                        bundle.tensors.total_bytes
                    );

                    for tensor in &bundle.tensors.tensors {
                        println!(
                            "  {} {:?} {} @ {}",
                            tensor.name, tensor.shape, tensor.dtype, tensor.offset
                        );
                    }
                }
                Err(e) => {
                    println!("Failed to read model bundle: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Run { text, model } => {
            println!("Running inference on '{}' with model {}", text, model);
            println!("ERROR: Offline runner not yet implemented");
            std::process::exit(1);
        }
    }

    Ok(())
}