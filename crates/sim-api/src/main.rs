use std::env;

use sim_api::http::HttpServerConfig;

#[tokio::main]
async fn main() {
    let mut args = env::args().skip(1);
    let mut bind_addr = "127.0.0.1:3000".to_string();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--addr" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --addr");
                    std::process::exit(2);
                };
                bind_addr = value;
            }
            _ => {
                bind_addr = arg;
            }
        }
    }

    let config = HttpServerConfig { bind_addr };
    if let Err(err) = sim_api::http::run(config).await {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}
